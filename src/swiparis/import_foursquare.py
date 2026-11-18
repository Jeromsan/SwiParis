"""Import TIST2015 TSV sans en-tête, mémoire bornée et étapes auditées."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import tempfile

SOURCE = "https://sites.google.com/site/yangdingqi/home/foursquare-dataset"
FORMAT_SOURCE = "https://huggingface.co/datasets/Davidham3/FourSquare_Global/blame/main/README.md"
DEFAULT_BBOX = (48.815, 2.224, 48.903, 2.470)  # south, west, north, east
MONTHS = {m: i for i, m in enumerate(
    "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split(), 1)}


def utc_time(value):
    # English month names parsed explicitly: independent of Windows locale.
    match = re.fullmatch(
        r"[A-Za-z]{3} ([A-Za-z]{3}) (\d{2}) (\d{2}):(\d{2}):(\d{2}) \+0000 (\d{4})", value)
    if not match:
        raise ValueError("expected TIST2015 UTC timestamp")
    month, day, hour, minute, second, year = match.groups()
    return datetime(int(year), MONTHS[month], int(day), int(hour), int(minute),
                    int(second), tzinfo=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def inside(lat, lon, bbox):
    south, west, north, east = bbox
    return south <= lat <= north and west <= lon <= east


def load_rows(path, columns, parser, insert, db, batch_size, stats):
    """Hash raw bytes, validate each row, retain only bounded error samples."""
    digest = hashlib.sha256()
    batch = []
    with path.open("rb") as stream:
        for number, raw in enumerate(stream, 1):
            digest.update(raw)
            stats["lines"] += 1
            if number % 1_000_000 == 0:
                print(f"{path.name}: {number:,} lines read", flush=True)
            try:
                fields = raw.decode("utf-8").rstrip("\r\n").split("\t")
                if len(fields) != columns or any(not f.strip() for f in fields):
                    raise ValueError(f"expected {columns} nonempty tab-separated columns")
                row = parser(fields)
            except (ValueError, KeyError, UnicodeError, OverflowError) as exc:
                stats["rejected"] += 1
                if len(stats["examples"]) < 10:
                    stats["examples"].append({"line": number, "reason": str(exc)})
                continue
            batch.append(row)
            stats["valid"] += 1
            if len(batch) >= batch_size:
                db.executemany(insert, batch)
                db.commit()
                batch.clear()
    db.executemany(insert, batch)
    db.commit()
    stats["sha256"] = digest.hexdigest()
    # A different release/header/delimiter must not silently yield an empty import.
    if not stats["valid"]:
        raise ValueError(f"No valid rows in {path}: {stats['examples']}")


def parse_poi(fields):
    venue, lat, lon, category, country = fields
    lat, lon = float(lat), float(lon)
    if not math.isfinite(lat) or not math.isfinite(lon) or not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError("invalid coordinates")
    if not re.fullmatch(r"[A-Z]{2}", country):
        raise ValueError("invalid ISO country code")
    return venue, lat, lon, category, country


def parse_checkin(fields):
    user, venue, timestamp, offset = fields
    offset = int(offset)
    if not -1440 < offset < 1440:
        raise ValueError("invalid timezone offset")
    return user, venue, utc_time(timestamp), offset


def volumes(db, relation, where="1=1"):
    # Separate counts let SQLite use the covering venue index instead of building
    # a huge DISTINCT venue B-tree while scanning the complete world history.
    return {key: db.execute(f"SELECT {expression} FROM {relation} WHERE {where}").fetchone()[0]
            for key, expression in (("visits", "count(*)"),
                                    ("visited_venues", "count(DISTINCT venue_id)"),
                                    ("users", "count(DISTINCT user_id)"))}


def run(args):
    for path in (args.pois, args.checkins):
        if not path.is_file():
            raise ValueError(f"Missing source file: {path}")
    if not args.inventory_only and args.categories is None:
        raise ValueError("Run --inventory-only, then provide --categories (JSON list of observed labels)")
    selected = []
    if args.categories:
        selected = json.loads(args.categories.read_text(encoding="utf-8-sig"))
        if not isinstance(selected, list) or not selected or any(not isinstance(c, str) or not c for c in selected):
            raise ValueError("Categories must be a nonempty JSON list of exact labels")
        selected = sorted(set(selected))
    if args.output.exists() or args.report.exists():
        raise ValueError("Output/report already exists; choose new paths (no overwrite)")
    if args.output.resolve() == args.report.resolve():
        raise ValueError("Database and report paths must differ")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    stats = {key: {"lines": 0, "valid": 0, "rejected": 0, "examples": []}
             for key in ("pois", "checkins")}
    report = {"source": {"dataset": "Foursquare Global-scale TIST2015", "url": SOURCE,
                         "format_reference": FORMAT_SOURCE,
                         "files": {"pois": str(args.pois.resolve()), "checkins": str(args.checkins.resolve())}},
              "parameters": {"min_checkins": args.min_checkins, "batch_size": args.batch_size,
                             "small_min_users": args.small_min_users, "small_min_visits": args.small_min_visits},
              "geographic_rule": {"type": "inclusive_bbox", "bbox_south_west_north_east": list(args.bbox),
                                  "country": "FR", "description": "Paris bounding rectangle; not municipal polygon"},
              "deduplication_key": ["user_id", "venue_id", "utc_time"], "input": stats,
              "inventory_only": args.inventory_only}
    # Staging on disk, beside the target; deleted even after failure. No global Python sets.
    with tempfile.TemporaryDirectory(prefix="foursquare-", dir=args.output.parent) as tmp:
        db = sqlite3.connect(str(Path(tmp) / "stage.sqlite"))
        try:
            db.executescript("""
                PRAGMA temp_store=FILE;
                PRAGMA cache_size=-131072;
                CREATE TABLE pois(venue_id TEXT PRIMARY KEY, latitude REAL NOT NULL,
                    longitude REAL NOT NULL, category TEXT NOT NULL, country TEXT NOT NULL);
                CREATE TABLE raw_visits(user_id TEXT, venue_id TEXT, utc_time TEXT, timezone_offset INTEGER);
            """)
            # First metadata occurrence wins, including conflicting duplicates (reported below).
            load_rows(args.pois, 5, parse_poi, "INSERT OR IGNORE INTO pois VALUES (?,?,?,?,?)",
                      db, args.batch_size, stats["pois"])
            unique = db.execute("SELECT count(*) FROM pois").fetchone()[0]
            stats["pois"]["duplicate_ids_first_wins"] = stats["pois"]["valid"] - unique
            south, west, north, east = args.bbox
            db.execute("CREATE TABLE paris AS SELECT * FROM pois WHERE country='FR' AND latitude BETWEEN ? AND ? AND longitude BETWEEN ? AND ?",
                       (south, north, west, east))
            categories = [{"category": c, "venues": n} for c, n in db.execute(
                "SELECT category,count(*) FROM paris GROUP BY category ORDER BY category")]
            print("Observed Paris categories:", json.dumps(categories, ensure_ascii=False), flush=True)
            report["categories_found"] = categories
            report["categories_retained"] = selected
            unknown = set(selected) - {item["category"] for item in categories}
            if unknown:
                raise ValueError(f"Categories absent from Paris metadata: {sorted(unknown)}")
            db.execute("CREATE TABLE selection(category TEXT PRIMARY KEY)")
            db.executemany("INSERT INTO selection VALUES (?)", [(c,) for c in selected])
            db.execute("CREATE TABLE leisure AS SELECT p.* FROM paris p JOIN selection s USING(category)")
            load_rows(args.checkins, 4, parse_checkin, "INSERT INTO raw_visits VALUES (?,?,?,?)",
                      db, args.batch_size, stats["checkins"])
            print("Building visit index and computing filter volumes...", flush=True)
            db.execute("CREATE INDEX raw_venue ON raw_visits(venue_id)")
            stages = {}
            stages["valid_input"] = volumes(db, "raw_visits")
            stages["valid_input"]["venues"] = unique
            for name, table in (("known_venue", "pois"), ("geography", "paris"), ("leisure", "leisure")):
                print(f"Counting stage: {name}", flush=True)
                all_known = name == "known_venue" and db.execute("""
                    SELECT 1 FROM (SELECT DISTINCT venue_id FROM raw_visits) r
                    LEFT JOIN pois p USING(venue_id) WHERE p.venue_id IS NULL LIMIT 1
                """).fetchone() is None
                stages[name] = (dict(stages["valid_input"]) if all_known else
                                volumes(db, "raw_visits", f"venue_id IN (SELECT venue_id FROM {table})"))
                stages[name]["venues"] = db.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            print("Deduplicating filtered visits...", flush=True)
            db.executescript("""
                CREATE TABLE dedup AS SELECT user_id, venue_id, utc_time, MIN(timezone_offset) AS timezone_offset
                  FROM raw_visits WHERE venue_id IN (SELECT venue_id FROM leisure)
                  GROUP BY user_id, venue_id, utc_time;
                CREATE INDEX dedup_user ON dedup(user_id,utc_time,venue_id);
            """)
            stages["deduplicated"] = volumes(db, "dedup")
            stages["deduplicated"]["venues"] = stages["deduplicated"]["visited_venues"]
            db.execute("CREATE TABLE eligible AS SELECT user_id FROM dedup GROUP BY user_id HAVING count(*)>=?", (args.min_checkins,))
            db.execute("CREATE UNIQUE INDEX eligible_user ON eligible(user_id)")
            stages["user_threshold"] = volumes(db, "dedup", "user_id IN (SELECT user_id FROM eligible)")
            stages["user_threshold"]["venues"] = stages["user_threshold"]["visited_venues"]
            report["stages"] = stages
            final = stages["user_threshold"]
            report["removed"] = {"unknown_venue_visits": stages["valid_input"]["visits"] - stages["known_venue"]["visits"],
                "geography_visits": stages["known_venue"]["visits"] - stages["geography"]["visits"],
                "category_visits": stages["geography"]["visits"] - stages["leisure"]["visits"],
                "duplicate_visits": stages["leisure"]["visits"] - stages["deduplicated"]["visits"],
                "user_threshold_visits": stages["deduplicated"]["visits"] - final["visits"]}
            report["warnings"] = []
            if args.inventory_only:
                report["warnings"].append("Inventory only: no leisure selection, no database published. Select exact categories and rerun.")
            else:
                if final["users"] < args.small_min_users or final["visits"] < args.small_min_visits:
                    report["warnings"].append(f"Small result: {final['visits']} visits, {final['venues']} venues, {final['users']} users; min_checkins remains {args.min_checkins}.")
                target = Path(tmp) / "result.sqlite"
                db.execute("ATTACH DATABASE ? AS result", (str(target),))
                db.executescript("""
                    CREATE TABLE result.venues(venue_id TEXT PRIMARY KEY, category TEXT NOT NULL,
                        latitude REAL NOT NULL, longitude REAL NOT NULL, country TEXT NOT NULL);
                    CREATE TABLE result.checkins(user_id TEXT NOT NULL, venue_id TEXT NOT NULL REFERENCES venues(venue_id),
                        utc_time TEXT NOT NULL, timezone_offset INTEGER NOT NULL,
                        PRIMARY KEY(user_id,utc_time,venue_id)) WITHOUT ROWID;
                    INSERT INTO result.venues SELECT venue_id,category,latitude,longitude,country FROM leisure
                        WHERE venue_id IN (SELECT venue_id FROM dedup WHERE user_id IN (SELECT user_id FROM eligible));
                    INSERT INTO result.checkins SELECT * FROM dedup WHERE user_id IN (SELECT user_id FROM eligible);
                    CREATE INDEX result.checkins_venue ON checkins(venue_id);
                """)
                assert db.execute("PRAGMA result.integrity_check").fetchone()[0] == "ok"
                assert not db.execute("PRAGMA result.foreign_key_check").fetchall()
                db.commit()
                db.execute("DETACH DATABASE result")
                target.rename(args.output)
            for name, query in (("valid_input", "SELECT min(utc_time),max(utc_time) FROM raw_visits"),
                                ("retained", "SELECT min(utc_time),max(utc_time) FROM dedup WHERE user_id IN (SELECT user_id FROM eligible)")):
                report.setdefault("period_utc", {})[name] = dict(zip(("start", "end"), db.execute(query).fetchone()))
        finally:
            db.close()
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"stages": report["stages"], "warnings": report["warnings"], "report": str(args.report)}, ensure_ascii=False, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkins", type=Path, required=True)
    parser.add_argument("--pois", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--categories", type=Path)
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--bbox", type=float, nargs=4, default=DEFAULT_BBOX, metavar=("S", "W", "N", "E"))
    parser.add_argument("--min-checkins", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=10000)
    parser.add_argument("--small-min-users", type=int, default=100)
    parser.add_argument("--small-min-visits", type=int, default=1000)
    args = parser.parse_args()
    s, w, n, e = args.bbox
    if not (-90 <= s < n <= 90 and -180 <= w < e <= 180):
        parser.error("Invalid bbox")
    if min(args.min_checkins, args.batch_size, args.small_min_users, args.small_min_visits) < 1:
        parser.error("Thresholds and batch size must be positive")
    try:
        run(args)
    except (ValueError, OSError, sqlite3.Error) as exc:
        parser.exit(2, f"Import failed: {exc}\n")


if __name__ == "__main__":
    main()
