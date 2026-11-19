"""Read-only catalog access shared by the local demo and application log checks."""
from contextlib import closing
import json
import math
import sqlite3

from .prepare_data import readonly
from .import_paris_data import parse_date, utc


def normalize_constraints(value):
    if not isinstance(value, dict) or set(value) - {"start_utc", "end_utc", "arrondissement", "budget_eur", "category"}:
        raise ValueError("Unknown constraint; use start_utc/end_utc/arrondissement/budget_eur/category")
    result = {}
    if bool(value.get("start_utc")) != bool(value.get("end_utc")):
        raise ValueError("Both start_utc and end_utc are required for a time slot")
    if value.get("start_utc"):
        start, end = parse_date(value["start_utc"]), parse_date(value["end_utc"])
        if end <= start:
            raise ValueError("End must follow start")
        result.update(start_utc=utc(start), end_utc=utc(end))
    if value.get("arrondissement") is not None:
        district = value["arrondissement"]
        if isinstance(district, bool) or not isinstance(district, int) or not 1 <= district <= 20:
            raise ValueError("Arrondissement must be an integer from 1 to 20")
        result["arrondissement"] = district
    if value.get("budget_eur") is not None:
        budget = value["budget_eur"]
        if isinstance(budget, bool) or not isinstance(budget, (int, float)) or not math.isfinite(budget) or budget < 0:
            raise ValueError("Budget must be a finite positive amount or zero")
        result["budget_eur"] = float(budget)
    if value.get("category"):
        if not isinstance(value["category"], str):
            raise ValueError("Category must be a string")
        result["category"] = value["category"]
    return result


class Catalog:
    def __init__(self, path):
        self.path = path
        with closing(readonly(path)) as db:
            self.manifest = json.loads(db.execute("SELECT value_json FROM metadata WHERE key='manifest'").fetchone()[0])
            self.categories = dict(db.execute("SELECT category,label FROM categories"))
        self.version = self.manifest["data_version"]

    def list_items(self, constraints=None, limit=200, offset=0):
        constraints = normalize_constraints(constraints or {})
        if constraints.get("category") and constraints["category"] not in self.categories:
            raise ValueError("Unknown common category")
        if offset < 0 or limit is not None and not 1 <= limit <= 500:
            raise ValueError("Invalid pagination")
        window = self.manifest["catalog_manifest"]
        start = constraints.get("start_utc", window["as_of_utc"])
        end = constraints.get("end_utc", window["window_end_utc"])
        where = ["i.source='paris_data'", "EXISTS(SELECT 1 FROM occurrences o WHERE o.item_id=i.item_id AND o.start_utc>=? AND o.end_utc<=?)"]
        params = [start, end]
        if "arrondissement" in constraints:
            # Source sessions are not linked to individual locations: avoid guessing.
            where.extend(["EXISTS(SELECT 1 FROM locations l WHERE l.item_id=i.item_id AND l.arrondissement=?)", "i.quality_flags_json NOT LIKE '%multi_location_schedule_unassigned%'"])
            params.append(constraints["arrondissement"])
        if "budget_eur" in constraints:
            where.append("i.price_amount_eur IS NOT NULL AND i.price_amount_eur<=? AND i.price_type!='conditional_free'")
            params.append(constraints["budget_eur"])
        if "category" in constraints:
            where.append("EXISTS(SELECT 1 FROM item_categories c WHERE c.item_id=i.item_id AND c.category=?)")
            params.append(constraints["category"])
        clause = " AND ".join(where)
        with closing(readonly(self.path)) as db:
            current = json.loads(db.execute("SELECT value_json FROM metadata WHERE key='manifest'").fetchone()[0])
            if current["data_version"] != self.version:
                raise ValueError("Catalog changed on disk; restart the service before logging more interactions")
            total = db.execute("SELECT count(*) FROM items i WHERE " + clause, params).fetchone()[0]
            query = "SELECT i.* FROM items i WHERE " + clause + " ORDER BY i.title COLLATE NOCASE,i.item_id"
            values = params[:]
            if limit is not None:
                query += " LIMIT ? OFFSET ?"
                values.extend((limit, offset))
            rows = []
            for raw in db.execute(query, values):
                item = dict(raw)
                item["quality_flags"] = json.loads(item.pop("quality_flags_json"))
                item["categories"] = [r[0] for r in db.execute("SELECT category FROM item_categories WHERE item_id=? ORDER BY category", (item["item_id"],))]
                item["locations"] = [dict(r) for r in db.execute("SELECT * FROM locations WHERE item_id=? ORDER BY location_index", (item["item_id"],))]
                item["next_occurrence"] = dict(db.execute("SELECT * FROM occurrences WHERE item_id=? AND start_utc>=? AND end_utc<=? ORDER BY start_utc LIMIT 1", (item["item_id"], start, end)).fetchone())
                rows.append(item)
        return dict(items=rows, total=total, offset=offset, limit=limit, catalog_version=self.version,
                    snapshot_window={k: window[k] for k in ("as_of_utc", "window_end_utc")}, categories=self.categories)

    def require_eligible(self, item_ids, constraints):
        eligible = {r["item_id"] for r in self.list_items(constraints, limit=None)["items"]}
        if not set(item_ids) <= eligible:
            raise ValueError("Unknown or ineligible catalog activity")
