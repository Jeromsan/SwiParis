"""Read-only catalog access shared by the local demo and application log checks."""
from contextlib import closing
from datetime import datetime, timezone
from functools import lru_cache
import json
import math
from zoneinfo import ZoneInfo

from .prepare_data import readonly
from .import_paris_data import parse_date, utc


PARIS = ZoneInfo("Europe/Paris")
PARIS_OFFSETS = (3600, 7200)


@lru_cache(maxsize=None)
def occurrence_utc(source_text):
    """Instant UTC d'une séance Paris Data, lu depuis l'heure locale de la source.

    La source écrit l'heure locale de Paris, mais avec +02:00 toute l'année :
    en novembre, « 14:00+02:00 » veut dire 14 h à Paris (13:00Z), pas 12:00Z.
    L'heure écrite est la bonne (elle correspond à ``date_description``), donc on
    la relit dans le fuseau Europe/Paris. Un décalage déjà juste, ou qui n'est
    pas un décalage de Paris, est gardé tel quel.
    """
    stated = datetime.fromisoformat(source_text.replace("Z", "+00:00"))
    offset = stated.utcoffset()
    local = stated.replace(tzinfo=PARIS)
    if offset is not None and (offset.total_seconds() not in PARIS_OFFSETS or local.utcoffset() == offset):
        return utc(stated.astimezone(timezone.utc))
    return utc(local.astimezone(timezone.utc))


# Compatibilité d'une séance avec le créneau demandé :
# - une séance courte (spectacle, atelier, visite guidée) doit tenir entièrement
#   dans le créneau, car on ne la suit pas à moitié ;
# - une séance de plus de 3 h correspond à des horaires d'ouverture (exposition,
#   parc, festival) : il suffit de pouvoir y passer au moins une heure.
# Exiger qu'une séance tienne entièrement dans le créneau excluait presque toutes
# les expositions : 2 activités seulement un mardi de 10 h à 13 h.
MIN_STAY_SECONDS = 3600
OPENING_HOURS_SECONDS = 3 * 3600
# Les heures comparées sont celles de occurrence_utc (heure locale de la source
# relue à Paris) et non les colonnes start_utc/end_utc, décalées d'une heure en hiver.
_SECONDS = "CAST(strftime('%s', {}) AS INTEGER)"
_START, _END = "occurrence_utc(o.source_start)", "occurrence_utc(o.source_end)"
SLOT_MATCH = (f"(({_START} >= ? AND {_END} <= ?) OR ("
              f"{_SECONDS.format(_END)} - {_SECONDS.format(_START)} > ?"
              f" AND {_SECONDS.format(f'min({_END}, ?)')} - {_SECONDS.format(f'max({_START}, ?)')} >= ?))")


def slot_params(start, end):
    """Paramètres de SLOT_MATCH, dans l'ordre des ?, pour le créneau [start, end)."""
    return [start, end, OPENING_HOURS_SECONDS, end, start, MIN_STAY_SECONDS]


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

    def list_items(self, constraints=None, limit=200, offset=0, item_ids=None):
        """Activités Paris Data éligibles aux contraintes ; ``item_ids`` restreint à ces activités."""
        constraints = normalize_constraints(constraints or {})
        if constraints.get("category") and constraints["category"] not in self.categories:
            raise ValueError("Unknown common category")
        if offset < 0 or limit is not None and not 1 <= limit <= 500:
            raise ValueError("Invalid pagination")
        window = self.manifest["catalog_manifest"]
        start = constraints.get("start_utc", window["as_of_utc"])
        end = constraints.get("end_utc", window["window_end_utc"])
        where = ["i.source='paris_data'", f"EXISTS(SELECT 1 FROM occurrences o WHERE o.item_id=i.item_id AND {SLOT_MATCH})"]
        params = slot_params(start, end)
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
        if item_ids is not None:
            item_ids = list(dict.fromkeys(item_ids))
            if not item_ids:
                where.append("0")
            else:
                where.append(f"i.item_id IN ({','.join('?' * len(item_ids))})")
                params.extend(item_ids)
        clause = " AND ".join(where)
        with closing(readonly(self.path)) as db:
            db.create_function("occurrence_utc", 1, occurrence_utc, deterministic=True)
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
                occurrence = dict(db.execute(
                    f"SELECT * FROM occurrences o WHERE o.item_id=? AND {SLOT_MATCH} ORDER BY {_START} LIMIT 1",
                    [item["item_id"], *slot_params(start, end)]).fetchone())
                occurrence.update(start_utc=occurrence_utc(occurrence["source_start"]),
                                  end_utc=occurrence_utc(occurrence["source_end"]))
                item["next_occurrence"] = occurrence
                rows.append(item)
        return dict(items=rows, total=total, offset=offset, limit=limit, catalog_version=self.version,
                    snapshot_window={k: window[k] for k in ("as_of_utc", "window_end_utc")}, categories=self.categories)

    def require_eligible(self, item_ids, constraints):
        eligible = {r["item_id"] for r in self.list_items(constraints, limit=None, item_ids=item_ids)["items"]}
        if not set(item_ids) <= eligible:
            raise ValueError("Unknown or ineligible catalog activity")
