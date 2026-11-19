"""Build the team's common SQLite file without changing source databases."""
import argparse
from collections import Counter
from contextlib import closing
import hashlib
from importlib.resources import files
import json
from pathlib import Path
import sqlite3
import tempfile

from .taxonomy import load_taxonomy, paris_categories


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def readonly(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError(f"Missing database: {path}")
    db = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    return db


def prepare(foursquare, paris_data, taxonomy_path, output, report_path, summary_path=None, force=False):
    taxonomy = load_taxonomy(taxonomy_path)
    inputs = [Path(p).resolve() for p in (foursquare, paris_data, taxonomy_path)]
    outputs = [Path(p).resolve() for p in (output, report_path, summary_path) if p is not None]
    if len(set(outputs)) != len(outputs) or set(inputs) & set(outputs):
        raise ValueError("Input/output paths must differ")
    for path in outputs:
        if path.exists() and not force:
            raise ValueError(f"Output exists: {path}; use --force to regenerate outputs")
        path.parent.mkdir(parents=True, exist_ok=True)
    manifest = dict(schema_version=1, taxonomy_version=taxonomy["version"],
                    taxonomy_sha256=sha256(taxonomy_path), foursquare_sha256=sha256(foursquare),
                    paris_data_sha256=sha256(paris_data))
    manifest["data_version"] = "common-" + hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()[:16]
    with tempfile.TemporaryDirectory(prefix="common-", dir=outputs[0].parent) as tmp:
        temp = Path(tmp) / "common.sqlite"
        with closing(readonly(foursquare)) as fs, closing(readonly(paris_data)) as pd, closing(sqlite3.connect(temp)) as db:
            db.executescript(files("swiparis").joinpath("common_schema.sql").read_text(encoding="utf-8"))
            db.executemany("INSERT INTO categories VALUES (?,?)", taxonomy["categories"].items())
            catalog_manifest = json.loads(pd.execute("SELECT manifest_json FROM snapshots").fetchone()[0])
            manifest["catalog_manifest"] = catalog_manifest
            for row in fs.execute("SELECT * FROM venues ORDER BY venue_id"):
                category = row["category"]
                if category not in taxonomy["foursquare"]:
                    raise ValueError(f"Foursquare label without explicit mapping: {category}")
                item = "foursquare:" + row["venue_id"]
                db.execute("INSERT INTO items VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                    item, "foursquare", row["venue_id"], category, "", "", None, category,
                    None, "unknown", None, None, "", None, "", None, "", "",
                    '["category_only_text","no_availability_or_price"]'))
                db.execute("INSERT INTO item_categories VALUES (?,?)", (item, taxonomy["foursquare"][category]))
                db.execute("INSERT INTO locations VALUES (?,?,?,?,?,?,?,?,?)", (item, 1, None, None, None, "Paris", None, row["latitude"], row["longitude"]))
            db.executemany("INSERT INTO visits VALUES (?,?,?,?,?)", (
                ("foursquare:" + str(r["user_id"]), str(r["user_id"]), "foursquare:" + r["venue_id"], r["utc_time"], r["timezone_offset"])
                for r in fs.execute("SELECT * FROM checkins ORDER BY user_id,utc_time,venue_id")))
            tags = {}
            for r in pd.execute("SELECT event_id,source_tag FROM event_tags"):
                tags.setdefault(r["event_id"], []).append(r["source_tag"])
            for r in pd.execute("SELECT * FROM events ORDER BY event_id"):
                db.execute("INSERT INTO items VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (
                    r["event_id"], "paris_data", r["source_id"], r["title"], r["description"], r["lead_text"],
                    r["source_url"], ";".join(sorted(tags.get(r["event_id"], []))), r["snapshot_id"],
                    r["price_type"], r["is_free"], r["price_amount_eur"], r["price_detail"], r["cover_url"],
                    r["cover_credit"], r["booking_url"], r["date_description"], r["audience"], r["quality_flags_json"]))
                db.executemany("INSERT INTO item_categories VALUES (?,?)", [(r["event_id"], c) for c in paris_categories(tags.get(r["event_id"], []), taxonomy)])
            db.executemany("INSERT INTO locations VALUES (?,?,?,?,?,?,?,?,?)", (
                (r["event_id"], r["location_index"], r["venue_name"], r["street"], r["postal_code"], r["city"], r["arrondissement"], r["latitude"], r["longitude"])
                for r in pd.execute("SELECT * FROM event_locations")))
            db.executemany("INSERT INTO occurrences VALUES (?,?,?,?,?)", (
                (r["event_id"], r["start_utc"], r["end_utc"], r["source_start"], r["source_end"])
                for r in pd.execute("SELECT * FROM event_occurrences")))
            db.execute("INSERT INTO metadata VALUES ('manifest',?)", (json.dumps(manifest, ensure_ascii=False),))
            db.commit()
            integrity = db.execute("PRAGMA integrity_check").fetchone()[0]
            if integrity != "ok" or db.execute("PRAGMA foreign_key_check").fetchall():
                raise ValueError("Common database failed integrity check")
            visits = db.execute("SELECT count(*) FROM visits").fetchone()[0]
            counts = dict(db.execute("SELECT category,count(*) FROM training_visits GROUP BY category"))
            venue_counts = dict(db.execute("SELECT category,count(*) FROM item_categories JOIN items USING(item_id) WHERE source='foursquare' GROUP BY category"))
            weights, assignments = Counter(), Counter()
            for r in db.execute("SELECT item_id,group_concat(category) FROM item_categories JOIN items USING(item_id) WHERE source='paris_data' GROUP BY item_id"):
                groups = r[1].split(",")
                assignments.update(groups)
                for c in groups:
                    weights[c] += 1 / len(groups)
            distribution = [dict(category=c, label=label, visits=counts.get(c, 0),
                                 visit_share_pct=round(100 * counts.get(c, 0) / visits, 3) if visits else 0,
                                 foursquare_venues=venue_counts.get(c, 0),
                                 paris_data_assignments=assignments[c],
                                 paris_data_weighted_events=round(weights[c], 4)) for c, label in taxonomy["categories"].items()]
            report = dict(manifest=manifest, taxonomy=taxonomy, category_distribution=distribution,
                          totals=dict(visits=visits, users=db.execute("SELECT count(DISTINCT user_id) FROM visits").fetchone()[0],
                                      foursquare_venues=sum(venue_counts.values()),
                                      paris_data_events=db.execute("SELECT count(*) FROM demo_items").fetchone()[0],
                                      occurrences=db.execute("SELECT count(*) FROM occurrences").fetchone()[0]),
                          zero_training_categories=[r["category"] for r in distribution if not r["visits"]],
                          sqlite_integrity=integrity)
        temp.replace(outputs[0])
    outputs[1].write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if summary_path:
        lines = ["# Données communes — rapport généré", "", f"Version : `{manifest['data_version']}`. Taxonomie v{taxonomy['version']}.", "",
                 f"{visits:,} visites Foursquare, {report['totals']['users']} utilisateurs ; {report['totals']['paris_data_events']} activités Paris Data.", "",
                 "| Catégorie | Visites | Part | Lieux Foursquare | Annonces Paris Data* |", "| --- | ---: | ---: | ---: | ---: |"]
        for r in distribution:
            lines.append(f"| {r['category']} | {r['visits']} | {r['visit_share_pct']:.2f} % | {r['foursquare_venues']} | {r['paris_data_assignments']} |")
        lines.extend(["", "*Une annonce Paris Data peut avoir plusieurs catégories ; les affectations ne sont pas additives. Le JSON contient aussi un décompte fractionnaire (1/n).",
                      "", "Les visites sont comptées après le filtrage Foursquare et le seuil de 10. La taxonomie ne crée aucune préférence ni correspondance d'identité entre les lieux historiques et les annonces actuelles.",
                      "", "Catégories sans visites d'entraînement : " + ", ".join(report["zero_training_categories"]) + ". Ne pas inventer d'historique pour les ateliers.", ""])
        Path(summary_path).write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"version": manifest["data_version"], "totals": report["totals"], "categories": distribution}, ensure_ascii=False, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--foursquare", type=Path, default=Path("data/processed/foursquare.sqlite"))
    parser.add_argument("--paris-data", type=Path, default=Path("data/processed/paris_data.sqlite"))
    parser.add_argument("--taxonomy", type=Path, default=Path("config/common_taxonomy.json"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/common.sqlite"))
    parser.add_argument("--report", type=Path, default=Path("outputs/common_report.json"))
    parser.add_argument("--summary", type=Path, default=Path("docs/data_summary.md"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        prepare(args.foursquare, args.paris_data, args.taxonomy, args.output, args.report, args.summary, args.force)
    except (OSError, ValueError, sqlite3.Error) as exc:
        parser.exit(2, f"Preparation failed: {exc}\n")


if __name__ == "__main__":
    main()
