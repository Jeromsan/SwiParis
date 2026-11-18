"""Nettoyage d'un export JSON Que Faire à Paris, sans appel réseau."""
import argparse
from collections import Counter
from contextlib import closing
from datetime import datetime, timedelta, timezone
import hashlib
from html import unescape
from html.parser import HTMLParser
from importlib.resources import files
import json
import math
from pathlib import Path
import re
import sqlite3
import tempfile
import unicodedata
from urllib.parse import urlsplit, urlunsplit

SOURCE_URL = "https://opendata.paris.fr/explore/dataset/que-faire-a-paris-/"
BBOX = (48.815, 2.224, 48.903, 2.470)
SCHEMA_VERSION = 1
NORMALIZATION_VERSION = 1


class TextParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.hidden = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self.hidden += 1
        if tag in ("p", "div", "br", "li", "h1", "h2", "h3", "section", "tr"):
            self.parts.append(" ")

    def handle_endtag(self, tag):
        if tag in ("script", "style") and self.hidden:
            self.hidden -= 1
        if tag in ("p", "div", "li", "h1", "h2", "h3", "section", "tr"):
            self.parts.append(" ")

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def text(value):
    if value is None:
        return ""
    parser = TextParser()
    parser.feed(str(value))
    parser.close()
    result = unicodedata.normalize("NFC", unescape("".join(parser.parts)))
    result = "".join(c if not unicodedata.category(c).startswith("C") else " " for c in result)
    return " ".join(result.split())


def parse_date(value):
    if not isinstance(value, str):
        raise ValueError("missing timestamp")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp without explicit UTC offset")
    return parsed.astimezone(timezone.utc)


def utc(value):
    return value.isoformat(timespec="seconds").replace("+00:00", "Z")


def url(value):
    if not isinstance(value, str):
        return None
    try:
        parsed = urlsplit(value.strip())
    except ValueError:
        return None
    if parsed.scheme.lower() not in ("https", "http") or not parsed.netloc:
        return None
    return value.strip()


def canonical_url(value):
    parsed = urlsplit(value)
    return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), parsed.path.rstrip("/"), "", ""))


def schedules(row, issues):
    """Explicit sessions win over the often very broad date_start/date_end envelope."""
    raw = row.get("occurrences")
    origin = "occurrences" if raw else "single_interval"
    if raw:
        if not isinstance(raw, str):
            issues["invalid_occurrences_field"] += 1
            return [], origin
        pairs = [part.split("_") for part in raw.split(";") if part.strip()]
    else:
        pairs = [[row.get("date_start"), row.get("date_end")]]
    found = {}
    for pair in pairs:
        try:
            if len(pair) != 2:
                raise ValueError("invalid session")
            start, end = map(parse_date, pair)
            if end <= start:
                raise ValueError("nonpositive session duration")
            # Even a dated envelope must not imply that a venue is open 24/7.
            if end - start > timedelta(hours=24):
                issues["interval_over_24h_not_usable_as_session"] += 1
                continue
        except (ValueError, TypeError, OverflowError):
            issues["invalid_session"] += 1
            continue
        key = (utc(start), utc(end))
        if key in found:
            issues["duplicate_session"] += 1
        else:
            found[key] = {"start_utc": key[0], "end_utc": key[1],
                          "source_start": pair[0], "source_end": pair[1]}
    return [found[k] for k in sorted(found)], origin


def locations(row, issues):
    source = row.get("locations")
    if not isinstance(source, list) or not source:
        point = row.get("lat_lon")
        source = [{"address_name": row.get("address_name"), "address_street": row.get("address_street"),
                   "address_zipCode": row.get("address_zipcode"), "address_city": row.get("address_city"),
                   "address_lat_lon": point, "location_type": "address"}]
    result = {}
    for entry in source:
        try:
            if not isinstance(entry, dict) or entry.get("location_type") not in ("address", "service", "lieu"):
                raise ValueError("nonphysical_location")
            point = entry.get("address_lat_lon")
            if isinstance(point, dict):
                lat, lon = float(point["lat"]), float(point["lon"])
            else:
                lat, lon = map(float, point.split(","))
            if not math.isfinite(lat) or not math.isfinite(lon) or not (-90 <= lat <= 90 and -180 <= lon <= 180) or (lat, lon) == (0, 0):
                raise ValueError("invalid_coordinates")
            south, west, north, east = BBOX
            if not (south <= lat <= north and west <= lon <= east):
                raise ValueError("outside_paris_bbox")
            postal = text(entry.get("address_zipCode"))
            city = text(entry.get("address_city"))
            district = int(postal[-2:]) if re.fullmatch(r"750(?:0[1-9]|1[0-9]|20)", postal) else 16 if postal == "75116" else None
            city_is_paris = bool(re.fullmatch(r"paris(?:\s+(?:france|0?[1-9]|1[0-9]|20))?", city.casefold()))
            if district is None and not city_is_paris:
                raise ValueError("no_paris_address")
            # A conflicting explicit suburb address is not rescued by a 75xxx postcode.
            if city and not city_is_paris:
                raise ValueError("conflicting_city")
            venue, street = text(entry.get("address_name")), text(entry.get("address_street"))
            if not venue and not street:
                raise ValueError("missing_physical_address")
        except (ValueError, TypeError, AttributeError, KeyError) as exc:
            reason = str(exc) if str(exc) in {"nonphysical_location", "invalid_coordinates", "outside_paris_bbox", "no_paris_address", "conflicting_city", "missing_physical_address"} else "invalid_coordinates"
            issues[reason] += 1
            continue
        key = (venue.casefold(), street.casefold(), postal, round(lat, 6), round(lon, 6))
        result[key] = dict(venue_name=venue, street=street, postal_code=postal, city="Paris",
                           arrondissement=district, latitude=lat, longitude=lon)
    return [result[k] for k in sorted(result)]


def source_id(row):
    value = row.get("id")
    return str(value).strip() if isinstance(value, (str, int)) and not isinstance(value, bool) else ""


def update_time(row):
    try:
        return parse_date(row.get("updated_at"))
    except (ValueError, TypeError, OverflowError):
        return datetime.min.replace(tzinfo=timezone.utc)


def prepare(rows, category_map, as_of, end):
    stages = Counter(input=len(rows))
    issues, location_issues = Counter(), Counter()
    rejections, accepted = [], []
    observed_tags = Counter()
    identities, urls, contents = {}, {}, {}

    def reject(number, row, reason, kept=None):
        rejections.append(dict(source_row=number, source_id=source_id(row) or None,
                               reason=reason, kept_source_id=kept))

    indexed = [(i, r) for i, r in enumerate(rows, 1)]
    # Prefer latest source update; ties keep the first input record deterministically.
    indexed.sort(key=lambda item: update_time(item[1]) if isinstance(item[1], dict) else datetime.min.replace(tzinfo=timezone.utc), reverse=True)
    for number, row in indexed:
        if not isinstance(row, dict):
            reject(number, {}, "invalid_record")
            continue
        identity, title, link = source_id(row), text(row.get("title")), url(row.get("url"))
        if not identity or not title or not link:
            reject(number, row, "missing_identity_title_or_source_url")
            continue
        stages["valid_identity"] += 1
        key_url = canonical_url(link)
        kept = identities.get(identity) or urls.get(key_url)
        if kept:
            reject(number, row, "duplicate_identity_or_url", kept)
            continue
        identities[identity], urls[key_url] = identity, identity
        stages["unique_identity"] += 1
        tags = sorted({text(tag) for tag in str(row.get("qfap_tags") or "").split(";") if text(tag)})
        observed_tags.update(tags)
        places = locations(row, location_issues)
        if not places:
            reject(number, row, "no_usable_paris_location")
            continue
        stages["paris_location"] += 1
        sessions, origin = schedules(row, issues)
        if not sessions:
            reject(number, row, "no_usable_schedule")
            continue
        stages["usable_schedule"] += 1
        window_sessions = [s for s in sessions if s["start_utc"] < utc(end) and s["end_utc"] > utc(as_of)]
        if not window_sessions:
            reject(number, row, "outside_time_window")
            continue
        stages["in_window"] += 1
        flags = []
        description = text(row.get("description"))
        if not description:
            flags.append("missing_description")
        groups = sorted({category_map[tag] for tag in tags if tag in category_map}) or ["autre"]
        if groups == ["autre"]:
            flags.append("no_mapped_activity_category")
        if any(p["arrondissement"] is None for p in places):
            flags.append("unknown_arrondissement")
        if len(places) > 1:
            flags.append("multi_location_schedule_unassigned")
        kind = {"gratuit": "free", "gratuit sous condition": "conditional_free", "payant": "paid"}.get(text(row.get("price_type")).casefold(), "unknown")
        detail = text(row.get("price_detail"))
        if kind == "conditional_free":
            flags.append("free_entry_requires_conditions")
        if kind in ("paid", "conditional_free", "unknown"):
            flags.append("numeric_price_unknown")
        updated = update_time(row)
        if updated.year == 1:
            flags.append("invalid_source_updated_at")
        event = dict(event_id=f"paris_data:{identity}", source_id=identity,
                     source_event_id=str(row["event_id"]) if row.get("event_id") is not None else None,
                     source_url=link, title=title, lead_text=text(row.get("lead_text")),
                     description=description, source_tags=tags, categories=groups, locations=places,
                     occurrences=window_sessions, price_type=kind,
                     is_free=1 if kind == "free" else 0 if kind == "paid" else None,
                     price_amount_eur=0.0 if kind == "free" else None, price_detail=detail,
                     date_description=text(row.get("date_description")),
                     source_date_start=row.get("date_start"), source_date_end=row.get("date_end"),
                     schedule_source=origin, updated_at_utc=utc(updated) if updated.year != 1 else None,
                     cover_url=url(row.get("cover_url")), cover_credit=text(row.get("cover_credit")),
                     booking_url=url(row.get("access_link")), access_type=text(row.get("access_type")),
                     audience=text(row.get("audience")), quality_flags=flags)
        # Conservative content duplicate: different dates or prices remain separate events.
        content = json.dumps([title.casefold(), description.casefold(),
                              [(p["venue_name"].casefold(), p["street"].casefold(), round(p["latitude"], 6), round(p["longitude"], 6)) for p in places],
                              [(s["start_utc"], s["end_utc"]) for s in window_sessions], kind, detail], ensure_ascii=False)
        if content in contents:
            reject(number, row, "duplicate_content", contents[content])
            continue
        contents[content] = identity
        accepted.append(event)
        stages["deduplicated"] += 1
    accepted.sort(key=lambda e: e["event_id"])
    # Include zero stages explicitly, for empty catalogs and machine consumers.
    for name in ("valid_identity", "unique_identity", "paris_location", "usable_schedule", "in_window", "deduplicated"):
        stages.setdefault(name, 0)
    assert len(accepted) + len(rejections) == len(rows)
    return accepted, rejections, dict(stages), dict(issues), dict(location_issues), dict(sorted(observed_tags.items()))


def run(args):
    as_of = (parse_date(args.as_of) if args.as_of else datetime.now(timezone.utc)).replace(microsecond=0)
    end = as_of + timedelta(days=args.days)
    output, report_path = args.output.resolve(), args.report.resolve()
    jsonl = output.with_suffix(".jsonl")
    if args.days < 1 or len({output, report_path, jsonl, args.source.resolve(), args.category_map.resolve()}) != 5:
        raise ValueError("Invalid window or colliding input/output paths")
    for path in (output, report_path, jsonl):
        if path.exists():
            raise ValueError(f"Output already exists: {path}; choose new paths")
    raw = args.source.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    rows = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a nonempty JSON export array (not an API records page)")
    if not isinstance(rows[0], dict) or not {"id", "title", "url", "occurrences", "date_start", "date_end", "locations"} <= rows[0].keys():
        raise ValueError("Export schema does not match the inspected Paris Data format")
    mapping_bytes = args.category_map.read_bytes()
    category_map = json.loads(mapping_bytes.decode("utf-8-sig"))
    if not isinstance(category_map, dict) or not all(isinstance(k, str) and isinstance(v, str) and k and v for k, v in category_map.items()):
        raise ValueError("Expected source tag -> normalized category JSON mapping")
    events, rejected, stages, issues, location_issues, tags = prepare(rows, category_map, as_of, end)
    schema = files("swiparis").joinpath("paris_data_schema.sql").read_text(encoding="utf-8")
    mapping_sha = hashlib.sha256(mapping_bytes).hexdigest()
    schema_sha = hashlib.sha256(schema.encode("utf-8")).hexdigest()
    version_inputs = json.dumps([sha, mapping_sha, schema_sha, utc(as_of), args.days, NORMALIZATION_VERSION])
    version_sha = hashlib.sha256(version_inputs.encode("utf-8")).hexdigest()
    snapshot = f"paris-data-{as_of.strftime('%Y%m%dT%H%M%SZ')}-{version_sha[:12]}"
    frozen = args.source.resolve().parent / "snapshots" / f"{sha}.json"
    frozen.parent.mkdir(parents=True, exist_ok=True)
    if frozen.exists():
        if hashlib.sha256(frozen.read_bytes()).hexdigest() != sha:
            raise ValueError("Existing frozen source has wrong hash")
    else:
        with frozen.open("xb") as stream:
            stream.write(raw)
    manifest = dict(snapshot_id=snapshot, schema_version=SCHEMA_VERSION,
                    normalization_version=NORMALIZATION_VERSION, source_url=SOURCE_URL,
                    source_path=str(args.source.resolve()), frozen_source=str(frozen),
                    source_sha256=sha, category_map=category_map,
                    category_map_sha256=mapping_sha, schema_sha256=schema_sha,
                    as_of_utc=utc(as_of), window_end_utc=utc(end), days=args.days,
                    geography=dict(bbox_south_west_north_east=BBOX, rule="physical Paris address AND inclusive bbox"),
                    time_policy="source explicit offsets preserved; UTC; overlap with [as_of, window_end)",
                    schedule_policy="explicit occurrences; else positive date interval <=24h; no recurrence inferred")
    occurrence_count = sum(len(e["occurrences"]) for e in events)
    report = dict(manifest=manifest, stages=stages, retained=dict(events=len(events),
                  occurrences=occurrence_count, location_rows=sum(len(e["locations"]) for e in events)),
                  rejected_by_reason=dict(Counter(r["reason"] for r in rejected)),
                  rejections=sorted(rejected, key=lambda r: r["source_row"]),
                  schedule_issues=issues, location_issues=location_issues,
                  observed_source_tags=tags,
                  retained_categories=dict(sorted(Counter(c for e in events for c in e["categories"]).items())),
                  retained_price_types=dict(Counter(e["price_type"] for e in events)),
                  quality_flags=dict(Counter(f for e in events for f in e["quality_flags"])),
                  outputs=dict(sqlite=str(output), clean_jsonl=str(jsonl), report=str(report_path)), warnings=[])
    if len(events) < 20:
        report["warnings"].append(f"Small catalog: {len(events)} events, {occurrence_count} sessions; filters unchanged")
    if any(e["schedule_source"] == "occurrences" for e in events):
        report["warnings"].append("UTC offsets are kept as supplied, including apparent inconsistencies with local date descriptions; verify listing before attending")
    output.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="paris-data-", dir=output.parent) as tmp:
        temp = Path(tmp)
        with closing(sqlite3.connect(temp / "catalog.sqlite")) as db:
            db.executescript(schema)
            db.execute("INSERT INTO snapshots VALUES (?,?,?,?,?,?,?)", (snapshot, SCHEMA_VERSION, SOURCE_URL, sha, utc(as_of), utc(end), json.dumps(manifest, ensure_ascii=False)))
            columns = ["event_id", "snapshot_id", "source_id", "source_event_id", "source_url", "title", "lead_text", "description", "price_type", "is_free", "price_amount_eur", "price_detail", "date_description", "source_date_start", "source_date_end", "schedule_source", "updated_at_utc", "cover_url", "cover_credit", "booking_url", "access_type", "audience", "quality_flags_json"]
            for event in events:
                e = dict(event, snapshot_id=snapshot, quality_flags_json=json.dumps(event["quality_flags"]))
                db.execute(f"INSERT INTO events ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})", [e[c] for c in columns])
                db.executemany("INSERT INTO event_locations VALUES (?,?,?,?,?,?,?,?,?)", [(e["event_id"], i, p["venue_name"], p["street"], p["postal_code"], p["city"], p["arrondissement"], p["latitude"], p["longitude"]) for i, p in enumerate(e["locations"], 1)])
                db.executemany("INSERT INTO event_occurrences VALUES (?,?,?,?,?)", [(e["event_id"], s["start_utc"], s["end_utc"], s["source_start"], s["source_end"]) for s in e["occurrences"]])
                db.executemany("INSERT INTO event_categories VALUES (?,?)", [(e["event_id"], c) for c in e["categories"]])
                db.executemany("INSERT INTO event_tags VALUES (?,?)", [(e["event_id"], t) for t in e["source_tags"]])
            db.executemany("INSERT INTO import_rejections VALUES (?,?,?,?)", [(r["source_row"], r["source_id"], r["reason"], r["kept_source_id"]) for r in rejected])
            db.commit()
            integrity = db.execute("PRAGMA integrity_check").fetchone()[0]
            fk = db.execute("PRAGMA foreign_key_check").fetchall()
            if integrity != "ok" or fk:
                raise ValueError(f"SQLite validation failed: {integrity}, {fk}")
            report["verification"] = dict(sqlite_integrity=integrity, foreign_key_errors=len(fk), accounted_source_records=len(events) + len(rejected))
        with (temp / "clean.jsonl").open("w", encoding="utf-8") as stream:
            for event in events:
                stream.write(json.dumps(dict(event, snapshot_id=snapshot), ensure_ascii=False) + "\n")
        (temp / "catalog.sqlite").rename(output)
        (temp / "clean.jsonl").rename(jsonl)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("stages", "retained", "rejected_by_reason", "retained_price_types", "quality_flags", "verification", "outputs", "warnings")}, ensure_ascii=False, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--category-map", type=Path, default=Path("config/paris_data_categories.json"))
    parser.add_argument("--as-of", help="Snapshot instant with explicit timezone; default: current UTC")
    parser.add_argument("--days", type=int, default=30)
    args = parser.parse_args()
    try:
        run(args)
    except (ValueError, TypeError, OSError, sqlite3.Error) as exc:
        parser.exit(2, f"Paris Data import failed: {exc}\n")


if __name__ == "__main__":
    main()
