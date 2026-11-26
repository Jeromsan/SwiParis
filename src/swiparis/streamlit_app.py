"""Interface Streamlit de SwiParis (démo locale, API FastAPI).

Parcours : disponibilité et filtres → jusqu'à 10 choix entre deux sorties → 3
recommandations, avec la persona à laquelle la personne ressemble. Les sorties gardées
vont dans « Ma sélection », enregistrée par l'API et liée à l'adresse de la page
(paramètre ``carnet``) : on la retrouve en revenant sur la même adresse, on peut en
retirer, annuler un retrait et l'exporter (agenda .ics, tableau .csv).
"""
from __future__ import annotations

import csv
from datetime import date, datetime, time, timedelta, timezone
import html
import io
import json
import os
from pathlib import Path
import re
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen
import uuid
from zoneinfo import ZoneInfo

import streamlit as st
import streamlit.components.v1 as components


API_URL = os.environ.get("SWIPARIS_API_URL", "http://127.0.0.1:8000")
# Adresse de l'API vue par le navigateur (lien « Voir l'annonce » qui passe par /go).
PUBLIC_API_URL = os.environ.get("SWIPARIS_PUBLIC_API_URL") or API_URL
MAX_ANSWERS = 10
PARIS = ZoneInfo("Europe/Paris")
CARNET = re.compile(r"[A-Za-z0-9_-]{8,64}")
DAYS = ("lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche")
MONTHS = ("janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août",
          "septembre", "octobre", "novembre", "décembre")
SEARCH_KEYS = ("session_id", "constraints", "pair", "ended", "recommendations", "answers", "stop_reason",
               "available_count", "pending_pair", "pending_pair_event", "recommendations-event",
               "interaction-event", "create-session-event", "reco-all", "choice_feedback")
API_MESSAGES = {
    "Session belongs to another catalog version": "La liste des activités a été mise à jour depuis le début de "
                                                  "ta recherche. Lance une nouvelle séance.",
    "Unknown session": "Cette séance n'existe plus. Lance une nouvelle séance.",
    "Catalog changed on disk": "La liste des activités a changé. Redémarre l'API, puis lance une nouvelle séance.",
    "Unknown or ineligible catalog activity": "Cette activité n'est plus disponible pour ce créneau.",
    "model_version does not match": "L'API a été relancée avec un autre réglage. Recharge la page.",
}


class ApiError(RuntimeError):
    """Erreur de l'API, déjà formulée pour l'utilisateur."""


def api(path: str, payload: dict | None = None) -> dict:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(API_URL + path, data=body,
                      headers={"Content-Type": "application/json"} if body else {})
    try:
        with urlopen(request, timeout=15) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        try:
            content = json.loads(exc.read().decode("utf-8"))
            detail = str(content.get("detail") or content.get("error") or "")
        except (ValueError, UnicodeError, AttributeError):
            detail = ""
        raise ApiError(friendly_error(exc.code, detail)) from exc
    except (URLError, TimeoutError, ConnectionError) as exc:
        raise ApiError("Le service SwiParis ne répond pas. Vérifie que l'API est lancée "
                       "(`.venv/bin/python -m uvicorn swiparis.api:app --port 8000`), puis recharge la page.") from exc


def friendly_error(code: int, detail: str) -> str:
    for start, text in API_MESSAGES.items():
        if detail.startswith(start):
            return text
    if code == 503:
        return "Le service démarre encore. Réessaie dans quelques secondes."
    return f"La demande n'a pas pu aboutir (erreur {code}). Détail technique : {detail or 'aucun'}."


def event_id(key: str) -> str:
    """Identifiant gardé jusqu'au succès : une requête rejouée n'est pas comptée deux fois."""
    pending = st.session_state.get(key)
    if not pending:
        pending = str(uuid.uuid4())
        st.session_state[key] = pending
    return pending


def carnet_id() -> str:
    """Identifiant de « Ma sélection », porté par l'adresse de la page (aucune donnée personnelle)."""
    value = st.query_params.get("carnet", "")
    if not CARNET.fullmatch(value or ""):
        value = uuid.uuid4().hex
        st.query_params["carnet"] = value
    return value


def load_selection(owner: str) -> list[dict]:
    return api("/api/saved?" + urlencode({"owner_id": owner}))["items"]


def inject_theme() -> None:
    st.markdown("""
<style>
:root { --ink:#152b2a; --muted:#61716e; --green:#167c67; --lime:#c9ef8a; --paper:#f4f7f4; }
[data-testid="stAppViewContainer"] { background:radial-gradient(ellipse at 15% 0%,#e5f3e9 0,transparent 38%),var(--paper); color:var(--ink); }
[data-testid="stHeader"] { background:rgba(244,247,244,.78); }
.main .block-container { max-width:1240px; padding:2.1rem 2.2rem 4rem; }
h1,h2,h3 { color:var(--ink); letter-spacing:-.035em; }
h1 { font-weight:780; }
p, [data-testid="stCaptionContainer"] { color:var(--muted); }
[data-testid="stVerticalBlockBorderWrapper"] {
  border:1px solid rgba(26,75,64,.12); border-radius:24px; background:rgba(255,255,255,.92);
  box-shadow:0 14px 38px rgba(23,59,48,.075); overflow:hidden;
  animation:cardArrive .48s cubic-bezier(.2,.75,.25,1) both;
  transition:transform .2s ease, box-shadow .2s ease, border-color .2s ease;
}
[data-testid="stVerticalBlockBorderWrapper"]:hover { transform:translateY(-3px); box-shadow:0 18px 44px rgba(23,59,48,.12); border-color:rgba(22,124,103,.28); }
[data-testid="stImage"] img { border-radius:16px; object-fit:cover; }
div[data-testid="stButton"]>button { border-radius:15px; min-height:3.1rem; font-weight:700; transition:transform .16s ease, box-shadow .16s ease, background .16s ease; }
div[data-testid="stButton"]>button:hover { transform:translateY(-2px); box-shadow:0 9px 20px rgba(22,124,103,.16); }
div[data-testid="stButton"]>button:active { transform:scale(.97); }
div[data-testid="stButton"]>button[kind="primary"] { border:0; color:#fff; background:linear-gradient(115deg,#167c67,#239b7a); }
[data-testid="stProgressBar"] > div > div { background:linear-gradient(90deg,#167c67,#a8d963); }
.swiparis-hero { padding:1.5rem 0 1rem; }
.swiparis-eyebrow { color:#167c67; font-size:.78rem; font-weight:800; letter-spacing:.14em; text-transform:uppercase; margin:0 0 .45rem; }
.swiparis-hero h1 { font-size:clamp(2.2rem,5vw,3.5rem); margin:.15rem 0 .55rem; line-height:1.02; }
.swiparis-hero p:last-child { font-size:1.08rem; max-width:720px; }
.choice-feedback { display:flex; gap:.75rem; align-items:center; margin:.3rem 0 1.1rem; padding:.9rem 1.15rem; border:1px solid #cce9d8; border-radius:16px; background:linear-gradient(100deg,#e7f7eb,#f5fae9); color:#175d4e; font-weight:700; animation:feedbackPop .5s cubic-bezier(.16,1,.3,1) both; }
.choice-feedback span { font-size:1.45rem; animation:heartPop .65s ease both; }
.sp-chip{display:inline-block;padding:.12rem .62rem;margin:0 .3rem .35rem 0;border-radius:999px;background:#e7f4ee;color:#175d4e;font-size:.8rem;font-weight:600}
.sp-chip.free{background:#e9f7d9;color:#2f6b12}
.sp-chip.warn{background:#fdf0dc;color:#8a4b06}
.sp-cover{width:100%;height:170px;object-fit:cover;border-radius:16px;display:block;margin-bottom:.6rem}
.sp-cover.small{height:120px}
.persona-box{margin:.4rem 0 1.2rem;padding:1rem 1.2rem;border:1px solid #cce9d8;border-radius:18px;background:linear-gradient(100deg,#eef8f1,#f7fbef)}
.persona-box b{color:#167c67}
.choice-index { color:#167c67; font-size:.75rem; font-weight:800; letter-spacing:.1em; text-transform:uppercase; margin:.15rem 0 .65rem; }
@keyframes cardArrive { from { opacity:0; transform:translateY(12px) scale(.992); } to { opacity:1; transform:translateY(0) scale(1); } }
@keyframes feedbackPop { from { opacity:0; transform:translateY(-8px) scale(.98); } to { opacity:1; transform:translateY(0) scale(1); } }
@keyframes heartPop { 0% { transform:scale(.55) rotate(-12deg); } 70% { transform:scale(1.2) rotate(5deg); } 100% { transform:scale(1) rotate(0); } }
@media (prefers-reduced-motion: reduce) { *,*::before,*::after { animation-duration:.01ms !important; transition-duration:.01ms !important; scroll-behavior:auto !important; } }
@media(max-width:700px) { .main .block-container { padding:1.2rem 1rem 3rem; } }
</style>
""", unsafe_allow_html=True)



def paris_iso(day: date, start_time: time, minutes: int) -> tuple[str, str]:
    start = datetime.combine(day, start_time, tzinfo=ZoneInfo("Europe/Paris"))
    end = start + timedelta(minutes=minutes)
    return (start.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
            end.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"))


def paris_slot_iso(day: date, slot: str, custom_start: time | None = None,
                   custom_minutes: int = 60) -> tuple[str, str]:
    """Convertit un choix de disponibilité en fenêtre stricte Europe/Paris."""
    windows = {
        "Toute la journée": (time(0, 0), 24 * 60),
        "Matin (6 h–12 h)": (time(6, 0), 6 * 60),
        "Après-midi (12 h–18 h)": (time(12, 0), 6 * 60),
        "Soirée (18 h–minuit)": (time(18, 0), 6 * 60),
    }
    if slot == "Horaire précis":
        if custom_start is None:
            raise ValueError("Choisis une heure de début")
        return paris_iso(day, custom_start, custom_minutes)
    if slot not in windows:
        raise ValueError("Créneau de disponibilité inconnu")
    return paris_iso(day, *windows[slot])



def show_map(items: list[dict]) -> None:
    config = api("/api/map-config")
    key = config.get("api_key", "")
    markers = []
    for item in items[:3]:
        for location in item.get("locations", [])[:1]:
            if location.get("latitude") is not None and location.get("longitude") is not None:
                markers.append({"lat": location["latitude"], "lng": location["longitude"],
                                "title": item["title"], "url": item.get("source_url", "")})
                break
    st.subheader("Carte des 3 recommandations")
    if not markers:
        st.caption("Aucune des trois sorties recommandées n'a de coordonnées cartographiques.")
        return
    if not key:
        st.caption("Aucune clé Google Maps n'est configurée : la carte interactive OpenStreetMap s'affiche. "
                   "Chaque repère propose aussi un lien Google Maps.")
        safe_markers = json.dumps(markers, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e")
        html_doc = f"""<!doctype html><html><head>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<style>html,body,#map{{height:100%;margin:0}} .leaflet-popup-content{{font:14px system-ui}}</style>
</head><body><div id="map" role="application" aria-label="Carte des recommandations"></div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script><script>
const markers={safe_markers};
const map=L.map('map').setView([48.8566,2.3522],12);
L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png',{{
  maxZoom:19,attribution:'&copy; OpenStreetMap contributors'
}}).addTo(map);
const bounds=[];
for(const m of markers){{
  const marker=L.marker([m.lat,m.lng]).addTo(map).bindTooltip(m.title);
  const popup=document.createElement('div');
  const title=document.createElement('strong'); title.textContent=m.title; popup.appendChild(title);
  const link=document.createElement('a');
  link.href='https://www.google.com/maps/search/?api=1&query='+encodeURIComponent(m.lat+','+m.lng);
  link.target='_blank'; link.rel='noopener'; link.textContent='Ouvrir dans Google Maps';
  popup.appendChild(document.createElement('br')); popup.appendChild(link);
  marker.bindPopup(popup); bounds.push([m.lat,m.lng]);
}}
if(bounds.length>1) map.fitBounds(bounds,{{padding:[24,24]}});
</script></body></html>"""
        components.html(html_doc, height=450, scrolling=False)
        return
    safe_markers = json.dumps(markers, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e")
    map_id = json.dumps(config.get("map_id", "DEMO_MAP_ID"))
    html_doc = f"""<!doctype html><html><body style="margin:0"><div id="map" style="height:440px;width:100%"></div>
<script>
const markers={safe_markers}; const mapId={map_id};
function initMap() {{
  const map = new google.maps.Map(document.getElementById('map'), {{ center: {{lat:48.8566,lng:2.3522}}, zoom:12, mapId }});
  const bounds = new google.maps.LatLngBounds();
  markers.forEach((m) => {{
    const marker = new google.maps.Marker({{position:{{lat:m.lat,lng:m.lng}},map,title:m.title}});
    bounds.extend(marker.position);
    const box = document.createElement('div');
    const title = document.createElement('strong'); title.textContent = m.title; box.appendChild(title);
    if (m.url && /^https?:\\/\\//i.test(m.url)) {{
      const link = document.createElement('a'); link.href = m.url; link.target = '_blank';
      link.rel = 'noopener'; link.textContent = 'Annonce source'; box.appendChild(document.createElement('br')); box.appendChild(link);
    }}
    const info = new google.maps.InfoWindow({{content:box}});
    marker.addListener('click', () => info.open({{anchor:marker,map}}));
  }});
  if (markers.length > 1) map.fitBounds(bounds);
}}
</script><script async src="https://maps.googleapis.com/maps/api/js?key={html.escape(key, quote=True)}&loading=async&callback=initMap"
onerror="document.getElementById('map').innerHTML='Google Maps est inaccessible. Vérifie la clé et ses restrictions.'"></script>
</body></html>"""
    components.html(html_doc, height=450, scrolling=False)



def parse_utc(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def paris_date(utc_text: str) -> date:
    return parse_utc(utc_text).astimezone(PARIS).date()


def day_label(moment: datetime) -> str:
    local = moment.astimezone(PARIS)
    return f"{DAYS[local.weekday()]} {local.day}{'er' if local.day == 1 else ''} {MONTHS[local.month - 1]}"


def hm(moment: datetime) -> str:
    return moment.astimezone(PARIS).strftime("%H:%M")


def plural(count: int, singular: str, several: str) -> str:
    return f"{count} {singular if count == 1 else several}"


def md(text: str) -> str:
    """Texte de la source affiché tel quel dans du Markdown."""
    return re.sub(r"([\\`*_{}\[\]<>#|~$])", r"\\\1", str(text))


def when(item: dict) -> str:
    occurrence = item.get("next_occurrence") or {}
    if not occurrence.get("start_utc"):
        return "Horaire à vérifier sur l'annonce"
    start, end = parse_utc(occurrence["start_utc"]), parse_utc(occurrence["end_utc"])
    day = day_label(start).capitalize()
    if end - start > timedelta(hours=3):
        return f"{day} · ouvert de {hm(start)} à {hm(end)}"
    return f"{day} · {hm(start)} – {hm(end)}"


def price(item: dict) -> str:
    kind, amount = item.get("price_type"), item.get("price_amount_eur")
    if kind == "free":
        return "Gratuit"
    if kind == "conditional_free":
        return "Gratuit sous conditions"
    if amount:
        text = f"{amount:.2f}".replace(".", ",").removesuffix(",00")
        return f"À partir de {text} €"
    if kind == "paid":
        return "Payant · prix sur l'annonce"
    return "Prix non renseigné"


def place(item: dict) -> str:
    location = (item.get("locations") or [{}])[0]
    district = location.get("arrondissement")
    area = f"Paris {district}{'er' if district == 1 else 'e'}" if district else (location.get("city") or "")
    return " · ".join(p for p in (location.get("venue_name"), area) if p) or "Lieu à vérifier sur l'annonce"


def address(item: dict) -> str:
    location = (item.get("locations") or [{}])[0]
    city = " ".join(p for p in (location.get("postal_code"), location.get("city")) if p)
    return ", ".join(p for p in (location.get("venue_name"), location.get("street"), city) if p)


def visit_window(item: dict) -> tuple[datetime, datetime] | None:
    """Moment de la sortie : la séance, ramenée au créneau choisi pour une ouverture longue."""
    occurrence = item.get("next_occurrence") or {}
    if not occurrence.get("start_utc"):
        return None
    start, end = parse_utc(occurrence["start_utc"]), parse_utc(occurrence["end_utc"])
    slot = item.get("slot") or {}
    if slot.get("start_utc") and slot.get("end_utc"):
        slot_start, slot_end = parse_utc(slot["start_utc"]), parse_utc(slot["end_utc"])
        if max(start, slot_start) < min(end, slot_end):
            start, end = max(start, slot_start), min(end, slot_end)
    return start, end


def criteria_text(constraints: dict) -> str:
    labels = st.session_state.get("labels", {})
    start, end = parse_utc(constraints["start_utc"]), parse_utc(constraints["end_utc"])
    parts = [f"{day_label(start).capitalize()}, {hm(start)} – {hm(end)}"]
    if constraints.get("arrondissement"):
        number = constraints["arrondissement"]
        parts.append(f"Paris {number}{'er' if number == 1 else 'e'}")
    if constraints.get("category"):
        parts.append(labels.get(constraints["category"], constraints["category"]))
    if constraints.get("budget_eur") is not None:
        parts.append(f"budget {constraints['budget_eur']:.0f} € max")
    return " · ".join(parts)



def ics_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace(";", "\\;").replace(",", "\\,").replace("\n", "\\n")


def ics_fold(line: str) -> str:
    """Coupe les lignes à 75 octets, comme l'exige le format iCalendar."""
    parts, current = [], ""
    for char in line:
        if len((current + char).encode("utf-8")) > 74:
            parts.append(current)
            current = " " + char
        else:
            current += char
    parts.append(current)
    return "\r\n".join(parts)


def ics_text(items: list[dict]) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    lines = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//SwiParis//Ma selection//FR", "CALSCALE:GREGORIAN",
             "METHOD:PUBLISH", "X-WR-CALNAME:SwiParis – Ma sélection"]
    for item in items:
        window = visit_window(item)
        if window is None:
            continue
        start, end = (moment.astimezone(timezone.utc) for moment in window)
        url = item.get("source_url") or ""
        notes = "\n".join(p for p in (price(item), url) if p)
        lines += ["BEGIN:VEVENT", f"UID:{item['item_id']}-{start:%Y%m%dT%H%M}@swiparis", f"DTSTAMP:{stamp}",
                  f"DTSTART:{start:%Y%m%dT%H%M%SZ}", f"DTEND:{end:%Y%m%dT%H%M%SZ}",
                  "SUMMARY:" + ics_escape(item.get("title") or "Sortie SwiParis"),
                  "LOCATION:" + ics_escape(address(item) or place(item)),
                  "DESCRIPTION:" + ics_escape(notes)]
        if url.startswith(("https://", "http://")):
            lines.append("URL:" + url)
        lines.append("END:VEVENT")
    lines.append("END:VCALENDAR")
    return "\r\n".join(ics_fold(line) for line in lines) + "\r\n"


def csv_text(items: list[dict]) -> str:
    labels = st.session_state.get("labels", {})
    buffer = io.StringIO()
    writer = csv.writer(buffer, delimiter=";")
    writer.writerow(["Titre", "Quand", "Lieu", "Adresse", "Prix", "Catégories", "Lien", "Gardée le"])
    for item in items:
        saved = item.get("saved_at_utc")
        writer.writerow([item.get("title", ""), when(item), place(item), address(item), price(item),
                         ", ".join(labels.get(c, c) for c in item.get("categories", [])),
                         item.get("source_url") or "",
                         parse_utc(saved).astimezone(PARIS).strftime("%d/%m/%Y %H:%M") if saved else ""])
    return buffer.getvalue()



def flash(kind: str, text: str) -> None:
    st.session_state.flash = (kind, text)


def clear_picks(prefix: str) -> None:
    for key in list(st.session_state.keys()):
        if isinstance(key, str) and key.startswith(prefix):
            st.session_state[key] = False


def keep(entries: list[dict]) -> None:
    """Ajoute une ou plusieurs sorties à « Ma sélection »."""
    try:
        result = api("/api/saved", {"event_id": str(uuid.uuid4()), "owner_id": st.session_state.carnet,
                                    "session_id": st.session_state.session_id, "items": entries})
    except ApiError as exc:
        flash("error", str(exc))
        return
    added = sum(entry["status"] != "already_saved" for entry in result["items"])
    already = len(result["items"]) - added
    text = (f"{plural(added, 'sortie ajoutée', 'sorties ajoutées')} à ta sélection." if added
            else "Ces sorties étaient déjà dans ta sélection.")
    if added and already:
        text += f" {plural(already, 'y était', 'y étaient')} déjà."
    flash("success", text + f" Tu as maintenant {plural(result['count'], 'sortie', 'sorties')} dans « Ma sélection ».")
    st.session_state.undo = None
    clear_picks("pick-reco-")
    st.session_state["reco-all"] = False


def remove(item_ids: list[str]) -> None:
    """Retire des sorties de « Ma sélection » ; le retrait peut être annulé."""
    try:
        result = api("/api/saved/remove", {"event_id": str(uuid.uuid4()), "owner_id": st.session_state.carnet,
                                           "item_ids": item_ids})
    except ApiError as exc:
        flash("error", str(exc))
        return
    st.session_state.undo = result["removed"] or None
    st.session_state.confirm_remove = None
    clear_picks("pick-sel-")
    st.session_state["sel-all"] = False
    if not result["removed"]:
        flash("info", "Ces sorties n'étaient déjà plus dans ta sélection.")


def restore(item_ids: list[str]) -> None:
    try:
        result = api("/api/saved/restore", {"event_id": str(uuid.uuid4()), "owner_id": st.session_state.carnet,
                                            "item_ids": item_ids})
    except ApiError as exc:
        flash("error", str(exc))
        return
    st.session_state.undo = None
    flash("success", f"{plural(len(result['restored']), 'sortie remise', 'sorties remises')} dans ta sélection.")



def go(page: str) -> None:
    st.session_state.page = page


def new_search() -> None:
    for key in SEARCH_KEYS:
        st.session_state.pop(key, None)
    clear_picks("pick-reco-")
    st.session_state.page = "search"


def toggle_all(keys: list[str], source: str) -> None:
    for key in keys:
        st.session_state[key] = st.session_state[source]



def cover(item: dict, small: bool = False) -> None:
    url = item.get("cover_url") or ""
    if url.startswith("https://"):
        st.markdown(f"<img class='sp-cover{' small' if small else ''}' src='{html.escape(url, quote=True)}' "
                    f"alt='' loading='lazy'>", unsafe_allow_html=True)


def chips(item: dict, extra: list[tuple[str, str]] = ()) -> None:
    labels = st.session_state.get("labels", {})
    values = [(labels.get(c, c), "") for c in item.get("categories", [])]
    values.append((price(item), "free" if item.get("price_type") == "free" else ""))
    values.extend(extra)
    st.markdown("".join(f"<span class='sp-chip {css}'>{html.escape(text)}</span>" for text, css in values),
                unsafe_allow_html=True)


def details(item: dict, heading: str = "####", extra: list[tuple[str, str]] = ()) -> None:
    st.markdown(f"{heading} {md(item.get('title') or 'Activité')}")
    chips(item, extra)
    st.markdown(f"🗓️ {md(when(item))}  \n📍 {md(place(item))}")
    lead = (item.get("lead_text") or "").strip()
    if lead:
        st.caption(md(lead[:240] + ("…" if len(lead) > 240 else "")))
    with st.expander("En savoir plus"):
        if item.get("description"):
            text = item["description"]
            st.write(md(text[:900] + ("…" if len(text) > 900 else "")))
        for label, value in (("Adresse", address(item)), ("Public", item.get("audience")),
                             ("Dates", item.get("date_description")), ("Tarif", item.get("price_detail"))):
            if value:
                st.markdown(f"**{label}** : {md(value)}")
        st.caption("Horaires et tarifs à confirmer sur l'annonce officielle.")


def keep_button(item: dict, saved_ids: set[str], key: str, entry: dict) -> None:
    if item["item_id"] in saved_ids:
        st.button("★ Dans ta sélection · Retirer", key=key, use_container_width=True,
                  on_click=remove, args=([item["item_id"]],),
                  help="Retire cette sortie de « Ma sélection ». Tu pourras annuler.")
    else:
        st.button("☆ Garder", key=key, use_container_width=True, on_click=keep, args=([entry],),
                  help="Ajoute cette sortie à « Ma sélection » pour la retrouver plus tard.")



def show_flash() -> None:
    message = st.session_state.pop("flash", None)
    if message:
        kind, text = message
        {"success": st.success, "error": st.error}.get(kind, st.info)(text)



def selection_page(items: list[dict]) -> None:
    st.title("Ma sélection")
    st.caption("Les sorties que tu as gardées. Elles restent enregistrées : reviens sur cette adresse "
               "(ajoute-la à tes favoris) pour les retrouver.")
    undo = st.session_state.get("undo")
    if undo:
        with st.container(border=True):
            a, b = st.columns([4, 1], vertical_alignment="center")
            a.info(f"{plural(len(undo), 'sortie retirée', 'sorties retirées')} de ta sélection.")
            b.button("Annuler", type="primary", use_container_width=True, on_click=restore, args=(undo,))
    if not items:
        st.info("Ta sélection est vide pour l'instant. Pendant une recherche, coche des sorties puis « Garder » : "
                "elles apparaîtront ici.")
        st.button("Trouver une sortie", type="primary", on_click=go, args=("search",))
        return

    now = datetime.now(timezone.utc)
    ordered = sorted(items, key=lambda i: (visit_window(i) is None, visit_window(i) or (now, now), i.get("title", "")))
    keys = [f"pick-sel-{item['item_id']}" for item in ordered]
    picked = [item for item in ordered if st.session_state.get(f"pick-sel-{item['item_id']}")]
    target, scope = (picked, f"les {len(picked)} cochées" if len(picked) > 1 else "la cochée") if picked \
        else (ordered, "toute la sélection")
    st.markdown(f"**{plural(len(items), 'sortie gardée', 'sorties gardées')}**")
    with st.container(border=True):
        a, b, c, d = st.columns([1.1, 1.8, 2, 2], vertical_alignment="center")
        a.checkbox("Tout cocher", key="sel-all", on_change=toggle_all, args=(keys, "sel-all"))
        b.button(f"Retirer ({len(picked)})", use_container_width=True, disabled=not picked,
                 on_click=lambda ids: st.session_state.update(confirm_remove=ids),
                 args=([i["item_id"] for i in picked],), help="Retire les sorties cochées. Tu pourras annuler.")
        c.download_button(f"📅 Agenda : {scope}", data=ics_text(target), file_name="swiparis-ma-selection.ics",
                          mime="text/calendar", use_container_width=True,
                          help="Fichier à ouvrir avec ton agenda (Apple Calendrier, Google Agenda, Outlook).")
        d.download_button(f"⬇️ Tableau : {scope}", data=csv_text(target).encode("utf-8-sig"),
                          file_name="swiparis-ma-selection.csv", mime="text/csv", use_container_width=True,
                          help="Fichier .csv à ouvrir dans Excel, Numbers ou Google Sheets.")
    confirm = st.session_state.get("confirm_remove")
    if confirm:
        with st.container(border=True):
            st.warning(f"Retirer {plural(len(confirm), 'sortie', 'sorties')} de ta sélection ? "
                       "Tu pourras annuler juste après.")
            a, b, _ = st.columns([1, 1, 3])
            a.button("Oui, retirer", type="primary", use_container_width=True, on_click=remove, args=(confirm,))
            b.button("Non, garder", use_container_width=True,
                     on_click=lambda: st.session_state.update(confirm_remove=None))

    current_day = None
    for item in ordered:
        window = visit_window(item)
        heading = day_label(window[0]).capitalize() if window else "Date à vérifier"
        if heading != current_day:
            st.subheader(heading)
            current_day = heading
        extra = []
        if not item.get("available", True):
            extra.append(("Plus dans la liste d'activités", "warn"))
        elif window and window[1] < now:
            extra.append(("Déjà passée", "warn"))
        with st.container(border=True):
            check, picture, body = st.columns([0.4, 1.4, 4])
            with check:
                st.checkbox(f"Cocher {item.get('title', '')}", key=f"pick-sel-{item['item_id']}",
                            label_visibility="collapsed")
            with picture:
                cover(item, small=True)
            with body:
                details(item, extra=extra)
                saved = item.get("saved_at_utc")
                if saved:
                    st.caption(f"Gardée le {parse_utc(saved).astimezone(PARIS):%d/%m/%Y à %H:%M}")
                left, right = st.columns(2)
                if (item.get("source_url") or "").startswith(("https://", "http://")):
                    left.link_button("Voir l'annonce ↗", item["source_url"], use_container_width=True)
                right.button("Retirer", key=f"remove-{item['item_id']}", use_container_width=True,
                             on_click=remove, args=([item["item_id"]],), help="Tu pourras annuler.")



def start_screen() -> None:
    st.markdown("""
<div class="swiparis-hero">
  <p class="swiparis-eyebrow">Sorties à Paris · recommandations personnalisées</p>
  <h1>Ta prochaine sortie<br>commence ici.</h1>
  <p>Dis-nous quand tu es libre. On te proposera deux idées à la fois et on affinera la sélection au fil de tes choix.</p>
</div>
""", unsafe_allow_html=True)
    labels = st.session_state.get("labels", {})
    window = st.session_state.get("window") or {}
    with st.form("session-setup"):
        st.subheader("Quand et où ?")
        first, last = (paris_date(window["as_of_utc"]), paris_date(window["window_end_utc"])) if window else (None, None)
        default_day = date.today() if not window or first <= date.today() <= last else first
        day = st.date_input("Date de sortie", value=default_day, min_value=first, max_value=last, format="DD/MM/YYYY",
                            help="Seuls les jours couverts par la liste d'activités sont proposés.")
        if window:
            st.caption(f"Activités disponibles du {first:%d/%m/%Y} au {last:%d/%m/%Y}.")
        slot = st.selectbox("Disponibilité", (
            "Toute la journée", "Matin (6 h–12 h)", "Après-midi (12 h–18 h)",
            "Soirée (18 h–minuit)", "Horaire précis",
        ), help="Un spectacle ou un atelier doit tenir dans ce créneau. Une exposition ou un parc doit être "
                "ouvert au moins 1 h pendant ce créneau.")
        if slot == "Horaire précis":
            start_time = st.time_input("Début du créneau", value=time(14, 0), step=900)
            duration = st.selectbox("Durée disponible", (60, 90, 120, 180, 240, 360), index=3,
                                    format_func=lambda n: f"{n // 60} h {n % 60:02d}".replace(" h 00", " h"))
        else:
            start_time, duration = None, 60
        arrondissement = st.selectbox("Arrondissement", [None, *range(1, 21)],
                                      format_func=lambda n: "Tout Paris" if n is None
                                      else f"Paris {n}{'er' if n == 1 else 'e'}")
        category = st.selectbox("Type de sortie", [None, *sorted(labels, key=labels.get)],
                                format_func=lambda c: "Tous les types" if c is None else labels[c])
        budget = st.number_input("Budget maximum par personne (€)", min_value=0.0, value=None, step=5.0,
                                 placeholder="Sans limite",
                                 help="Avec un budget, les sorties dont le prix est inconnu sont écartées.")
        consent = st.checkbox("J'accepte que mes choix soient enregistrés de façon anonyme pour le projet SwiParis.")
        st.caption("Sont enregistrés : le créneau, tes choix entre deux sorties, les sorties affichées, ouvertes "
                   "et gardées. Aucun nom, e-mail ni position n'est demandé.")
        submitted = st.form_submit_button("Commencer", type="primary")

    if not submitted:
        return
    if not consent:
        st.error("Coche la case d'accord pour commencer : sans elle, rien ne peut être enregistré.")
        return
    start_utc, end_utc = paris_slot_iso(day, slot, start_time, int(duration))
    constraints = {"start_utc": start_utc, "end_utc": end_utc}
    if arrondissement is not None:
        constraints["arrondissement"] = arrondissement
    if category:
        constraints["category"] = category
    if budget is not None:
        constraints["budget_eur"] = float(budget)
    with st.spinner("On regarde ce qui est possible sur ce créneau…"):
        count = int(api("/api/catalog?" + urlencode({**constraints, "limit": 1}))["total"])
        if count < 6:
            st.warning(f"Seulement {plural(count, 'sortie correspond', 'sorties correspondent')} à ces critères. "
                       "Il en faut au moins six pour comparer deux par deux. Essaie une autre date, un créneau "
                       "plus large, « Tout Paris » ou « Tous les types ».")
            return
        health = api("/health")
        created = api("/api/sessions", {
            "event_id": event_id("create-session-event"), "consent": True,
            "constraints": constraints, "catalog_version": health["catalog_version"],
            "model_version": health["model_version"], "data_origin": "pilot",
        })
    for key in SEARCH_KEYS:
        st.session_state.pop(key, None)
    st.session_state.update(session_id=created["session_id"], constraints=constraints, pair=None, ended=False,
                            stop_reason=None, answers=0, available_count=count, recommendations=None)
    st.rerun()


def advance() -> None:
    """Prépare la paire suivante, ou termine la phase des choix."""
    if st.session_state.get("pair") is not None or st.session_state.get("ended"):
        return
    sid = st.session_state.session_id
    pending = st.session_state.get("pending_pair")
    if pending is None:
        suggestion = api("/next_pair?" + urlencode({"session_id": sid, "owner_id": st.session_state.carnet}))
        st.session_state.answers = suggestion.get("answers", 0)
        if suggestion.get("done"):
            st.session_state.ended = True
            st.session_state.stop_reason = suggestion.get("stop_reason")
            st.session_state.available_count = suggestion.get("eligible_count", 0)
            return
        pending = suggestion["comparison"]
        st.session_state.pending_pair = pending
        st.session_state.pending_pair_event = str(uuid.uuid4())
    result = api("/api/comparisons", {"event_id": st.session_state.pending_pair_event, "session_id": sid,
                                      "item_a": pending["item_a"], "item_b": pending["item_b"]})
    st.session_state.pop("pending_pair", None)
    st.session_state.pop("pending_pair_event", None)
    st.session_state.pair = result


def choose(choice: str) -> None:
    pair = st.session_state.pair
    try:
        api("/log_interaction", {"event_id": event_id("interaction-event"),
                                 "session_id": st.session_state.session_id,
                                 "comparison_id": pair["comparison_id"], "choice": choice})
    except ApiError as exc:
        flash("error", str(exc))
        return
    if choice == "skip":
        feedback = "On passe à une nouvelle comparaison."
    else:
        selected = pair["left"] if choice == "left" else pair["right"]
        feedback = f"Bien noté : {selected.get('title', 'ta sortie préférée')} — j’affine la suite."
    st.session_state.choice_feedback = feedback
    st.session_state.pop("interaction-event", None)
    st.session_state.pair = None


def pair_screen(saved_ids: set[str]) -> None:
    feedback = st.session_state.pop("choice_feedback", None)
    if feedback:
        st.markdown(f'<div class="choice-feedback"><span>✨</span><div>{html.escape(feedback)}</div></div>',
                    unsafe_allow_html=True)
    pair = st.session_state.pair
    answers = st.session_state.get("answers", 0)
    st.title("Laquelle choisirais-tu ?")
    st.caption("Il n'y a pas de bonne réponse. Chaque choix affine tes recommandations.")
    st.progress(min(answers / MAX_ANSWERS, 1.0), text=f"Choix {answers} sur {MAX_ANSWERS}")
    columns = st.columns(2, gap="large")
    for number, (side, column) in enumerate(zip(("left", "right"), columns), 1):
        item = pair[side]
        with column, st.container(border=True):
            st.markdown(f'<p class="choice-index">Option {number}</p>', unsafe_allow_html=True)
            cover(item)
            details(item, heading="###")
            if (item.get("source_url") or "").startswith(("https://", "http://")):
                st.link_button("Voir l'annonce ↗", item["source_url"], use_container_width=True)
            st.button("💚  Je préfère celle-ci", key=f"choose-{side}", type="primary", use_container_width=True,
                      on_click=choose, args=(side,))
            keep_button(item, saved_ids, f"pairkeep-{side}", {"item_id": item["item_id"]})
    st.button("Passer cette comparaison", key="skip-pair", on_click=choose, args=("skip",),
              help="Ce choix ne compte pas : une autre paire te sera proposée.")


def persona_box(persona: dict | None) -> None:
    """Persona(s) à laquelle la personne ressemble, d'après ses choix."""
    if not persona or not persona.get("top"):
        return
    top = persona["top"]
    names = " · ".join(f"<b>{html.escape(p['name'])}</b> ({p['share']:.0%})" for p in top)
    used = persona.get("from_selection", 0)
    basis = (f"Tes choix et les {used} sorties de ta sélection ressemblent le plus à" if used
             else "Tes choix ressemblent le plus à")
    st.markdown(f"<div class='persona-box'>{basis} : {names}</div>", unsafe_allow_html=True)
    with st.expander("Comment c'est calculé ?"):
        st.write("Les personas sont des groupes d'habitudes de sortie observés dans les données Foursquare "
                 "(catégories, moment de la journée, quartiers), pas des profils inventés. Après chaque choix, on "
                 "calcule à quelles personas tes réponses ressemblent ; le pourcentage indique cette ressemblance. "
                 "Si tu as déjà gardé au moins deux sorties, ta sélection sert de point de départ. "
                 "Le classement tient ensuite compte de tes choix eux-mêmes.")
        for p in top:
            st.markdown(f"**{md(p['name'])}** — {md(p['description'])}")


def results_screen(saved_ids: set[str]) -> None:
    sid = st.session_state.session_id
    if st.session_state.get("recommendations") is None:
        with st.spinner("On prépare tes recommandations…"):
            result = api("/recommendations", {"event_id": event_id("recommendations-event"), "session_id": sid,
                                              "owner_id": st.session_state.carnet})
        st.session_state.pop("recommendations-event", None)
        st.session_state.recommendations = result
    recommendations = st.session_state.recommendations
    items = recommendations.get("items", [])
    answers = st.session_state.get("answers", 0)
    st.title("Tes sorties recommandées")
    st.caption(f"{criteria_text(st.session_state.constraints)} · classement personnalisé à partir de "
               f"{plural(answers, 'choix', 'choix')}.")
    reason = st.session_state.get("stop_reason")
    if reason == "no_informative_pair":
        st.info(f"La comparaison s'est arrêtée après {plural(answers, 'choix', 'choix')} : il ne restait pas de "
                "paire assez différente sur ce créneau. Le classement utilise les choix déjà faits.")
    elif reason == "max_offers":
        st.info("La séance a atteint sa limite de comparaisons. Le classement utilise les choix enregistrés.")
    persona_box(recommendations.get("persona"))
    if not items:
        st.info("Toutes les sorties de ce créneau ont été écartées pendant les choix : il n'en reste pas à "
                "proposer. Lance une nouvelle séance avec un créneau plus large ou moins de filtres.")
        st.button("Nouvelle séance", type="primary", on_click=new_search)
        return
    selectable = [item for item in items if item["item_id"] not in saved_ids]
    keys = [f"pick-reco-{item['item_id']}" for item in selectable]
    picked = [item for item in selectable if st.session_state.get(f"pick-reco-{item['item_id']}")]
    with st.container(border=True):
        a, b, c = st.columns([1.1, 2, 2], vertical_alignment="center")
        a.checkbox("Tout cocher", key="reco-all", on_change=toggle_all, args=(keys, "reco-all"), disabled=not keys)
        b.button(f"☆ Garder les {len(picked)} sorties cochées" if len(picked) > 1 else "☆ Garder la sortie cochée",
                 type="primary", use_container_width=True, disabled=not picked, on_click=keep,
                 args=([{"item_id": i["item_id"], "impression_id": i["impression_id"]} for i in picked],),
                 help="Coche une ou plusieurs sorties ci-dessous, puis garde-les en une fois.")
        c.button(f"Voir ma sélection ({len(saved_ids)})", use_container_width=True, on_click=go, args=("selection",))
    for index, item in enumerate(items):
        if index < 2:
            st.subheader("Meilleur match" if index == 0 else "Autres sorties recommandées")
        is_saved = item["item_id"] in saved_ids
        with st.container(border=True):
            if item.get("selected_by_user"):
                st.success("Tu avais choisi cette sortie pendant les comparaisons.")
            check, body = st.columns([0.35, 4])
            with check:
                if not is_saved:
                    st.checkbox(f"Cocher {item.get('title', '')}", key=f"pick-reco-{item['item_id']}",
                                label_visibility="collapsed")
            with body:
                cover(item)
                details(item, heading="###", extra=[("★ Dans ta sélection", "free")] if is_saved else [])
                left, right = st.columns(2)
                left.link_button("Voir l'annonce ↗", f"{PUBLIC_API_URL}/go/{quote(item['impression_id'])}",
                                 use_container_width=True)
                with right:
                    keep_button(item, saved_ids, f"keep-{item['item_id']}",
                                {"item_id": item["item_id"], "impression_id": item["impression_id"]})
    show_map(items)


def search_page(saved_ids: set[str]) -> None:
    if not st.session_state.get("session_id"):
        start_screen()
        return
    advance()
    if not st.session_state.get("ended"):
        pair_screen(saved_ids)
    else:
        results_screen(saved_ids)


def sidebar(count: int) -> None:
    with st.sidebar:
        st.markdown("## 🗼 SwiParis")
        st.caption("Des sorties à Paris choisies selon tes goûts.")
        st.radio("Navigation", ("search", "selection"), key="page", label_visibility="collapsed",
                 format_func=lambda p: "🔎 Trouver une sortie" if p == "search" else f"★ Ma sélection ({count})")
        if st.session_state.get("session_id"):
            st.button("Nouvelle séance", use_container_width=True, on_click=new_search,
                      help="Recommence avec d'autres critères. Ta sélection est conservée.")
        st.divider()
        st.caption("Ta sélection est enregistrée et liée à l'adresse de cette page. Ajoute la page à tes favoris "
                   "pour la retrouver plus tard.")


def render() -> None:
    st.set_page_config(page_title="SwiParis", page_icon="🗼", layout="wide")
    inject_theme()
    st.session_state.setdefault("page", "search")
    try:
        st.session_state.carnet = carnet_id()
        if "labels" not in st.session_state:
            meta = api("/api/catalog?limit=1")
            st.session_state.labels = meta.get("categories", {})
            st.session_state.window = meta.get("snapshot_window") or {}
        saved = load_selection(st.session_state.carnet)
    except ApiError as exc:
        st.title("SwiParis")
        st.error(str(exc))
        st.button("Réessayer", type="primary")
        return
    sidebar(len(saved))
    show_flash()
    try:
        if st.session_state.page == "selection":
            selection_page(saved)
        else:
            search_page({item["item_id"] for item in saved})
    except ApiError as exc:
        st.error(str(exc))
        st.button("Réessayer", type="primary")


def main() -> None:
    """Lance Streamlit depuis la commande installée ``swiparis-streamlit``."""
    from streamlit.web import cli as stcli
    app_file = str(Path(__file__).resolve())
    sys.argv = ["streamlit", "run", app_file]
    stcli.main()


if __name__ == "__main__":
    render()
