"""Streamlit front end for the local SwiParis FastAPI demo."""
from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
import html
import json
import os
from pathlib import Path
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import uuid
from zoneinfo import ZoneInfo

import streamlit as st
import streamlit.components.v1 as components


API_URL = os.environ.get("SWIPARIS_API_URL", "http://127.0.0.1:8000")


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
.choice-index { color:#167c67; font-size:.75rem; font-weight:800; letter-spacing:.1em; text-transform:uppercase; margin:.15rem 0 .65rem; }
@keyframes cardArrive { from { opacity:0; transform:translateY(12px) scale(.992); } to { opacity:1; transform:translateY(0) scale(1); } }
@keyframes feedbackPop { from { opacity:0; transform:translateY(-8px) scale(.98); } to { opacity:1; transform:translateY(0) scale(1); } }
@keyframes heartPop { 0% { transform:scale(.55) rotate(-12deg); } 70% { transform:scale(1.2) rotate(5deg); } 100% { transform:scale(1) rotate(0); } }
@media (prefers-reduced-motion: reduce) { *,*::before,*::after { animation-duration:.01ms !important; transition-duration:.01ms !important; scroll-behavior:auto !important; } }
@media(max-width:700px) { .main .block-container { padding:1.2rem 1rem 3rem; } }
</style>
""", unsafe_allow_html=True)


def api(path: str, payload: dict | None = None) -> dict:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(API_URL + path, data=body,
                      headers={"Content-Type": "application/json"} if body else {})
    try:
        with urlopen(request, timeout=8) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        try:
            detail = json.loads(exc.read().decode("utf-8")).get("detail", "Erreur API")
        except (ValueError, UnicodeError):
            detail = "Erreur API"
        raise RuntimeError(f"API {exc.code}: {detail}") from exc
    except (URLError, TimeoutError) as exc:
        raise RuntimeError("API indisponible. Lance d'abord `swiparis-api`.") from exc


def event_id(key: str) -> str:
    pending = st.session_state.get(key)
    if not pending:
        pending = str(uuid.uuid4())
        st.session_state[key] = pending
    return pending


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


def start_screen() -> None:
    st.markdown("""
<div class="swiparis-hero">
  <p class="swiparis-eyebrow">Sorties à Paris · recommandations personnalisées</p>
  <h1>Ta prochaine sortie<br>commence ici.</h1>
  <p>Dis-nous quand tu es libre. On te proposera deux idées à la fois et on affinera la sélection au fil de tes choix.</p>
</div>
""", unsafe_allow_html=True)
    health = api("/health")
    catalog = api("/api/catalog?limit=1")
    category_labels = catalog.get("categories", {})

    with st.form("session-setup"):
        st.subheader("Quand et où ?")
        day = st.date_input("Date de sortie", value=date.today())
        slot = st.selectbox("Disponibilité", (
            "Toute la journée", "Matin (6 h–12 h)", "Après-midi (12 h–18 h)",
            "Soirée (18 h–minuit)", "Horaire précis",
        ))
        if slot == "Horaire précis":
            start_time = st.time_input("Début du créneau", value=time(14, 0), step=900)
            duration = st.selectbox("Durée disponible", (60, 90, 120, 180, 240, 360),
                                    format_func=lambda n: f"{n} minutes")
        else:
            start_time, duration = None, 60
        arrondissement = st.selectbox("Arrondissement", [None, *range(1, 21)],
                                      format_func=lambda n: "Tous" if n is None else f"{n}e")
        category = st.selectbox("Catégorie", [None, *sorted(category_labels)],
                                format_func=lambda c: "Toutes" if c is None else category_labels[c])
        use_budget = st.checkbox("Ajouter un budget maximum")
        budget = st.number_input("Budget maximum par personne (€)", min_value=0.0, value=25.0, step=5.0,
                                 disabled=not use_budget)
        consent = st.checkbox("J'accepte de contribuer au pilote avec mes choix de séance.")
        submitted = st.form_submit_button("Commencer", type="primary")

    if not submitted:
        return
    if not consent:
        st.error("Le consentement est nécessaire avant de démarrer une séance du pilote.")
        return
    start_utc, end_utc = paris_slot_iso(day, slot, start_time, int(duration))
    constraints = {"start_utc": start_utc, "end_utc": end_utc}
    if arrondissement is not None:
        constraints["arrondissement"] = arrondissement
    if category:
        constraints["category"] = category
    if use_budget:
        constraints["budget_eur"] = float(budget)
    available = api("/api/catalog?" + urlencode({**constraints, "limit": 1}))
    count = int(available["total"])
    st.caption(f"{count} sortie(s) correspondent à ces filtres.")
    if count < 6:
        st.error("Il faut au moins six sorties disponibles pour permettre jusqu'à dix choix sans répéter "
                 "les paires. Essaie une autre date, un créneau plus large ou retire un filtre.")
        return
    created = api("/api/sessions", {
        "event_id": event_id("create-session-event"), "consent": True,
        "constraints": constraints, "catalog_version": health["catalog_version"],
        "model_version": health["model_version"], "data_origin": "pilot",
    })
    st.session_state.session_id = created["session_id"]
    st.session_state.constraints = constraints
    st.session_state.pair = None
    st.session_state.ended = False
    st.session_state.stop_reason = None
    st.session_state.answers = 0
    st.session_state.available_count = count
    st.session_state.recommendations = None
    for key in ("pending_pair", "pending_pair_event", "interaction-event", "recommendations-event"):
        st.session_state.pop(key, None)
    st.rerun()


def item_card(item: dict) -> None:
    if item.get("cover_url"):
        st.image(item["cover_url"], use_container_width=True)
    categories = " · ".join(item.get("categories", [])) or "Sortie à Paris"
    st.markdown(f'<p class="choice-index">{html.escape(categories)}</p>', unsafe_allow_html=True)
    st.markdown(f"### {item['title']}")
    if item.get("description"):
        st.write(item["description"][:550])
    if item.get("price_amount_eur") is None:
        price = "Prix non renseigné"
    elif item.get("price_type") == "free":
        price = "Gratuit"
    else:
        price = f"À partir de {item['price_amount_eur']:.0f} €"
    st.write(price)
    occurrence = item.get("next_occurrence") or {}
    if occurrence.get("source_start"):
        st.write(f"Séance : {occurrence['source_start']}")
    location = item.get("locations", [{}])[0]
    if location.get("city") or location.get("arrondissement"):
        place = location.get("city") or "Paris"
        if location.get("arrondissement"):
            place += f" · {location['arrondissement']}e arrondissement"
        st.write(place)
    if item.get("source_url"):
        st.link_button("Voir l'annonce source", item["source_url"], use_container_width=True)


def pair_screen() -> None:
    sid = st.session_state.session_id
    if st.session_state.get("pair") is None and not st.session_state.get("ended"):
        pending = st.session_state.get("pending_pair")
        if pending is None:
            suggestion = api(f"/next_pair?session_id={sid}")
            if suggestion.get("done"):
                st.session_state.ended = True
                st.session_state.stop_reason = suggestion.get("stop_reason")
                st.session_state.answers = suggestion.get("answers", 0)
                st.session_state.available_count = suggestion.get("eligible_count", 0)
                pending = None
            else:
                pending = suggestion["comparison"]
                st.session_state.pending_pair = pending
                st.session_state.pending_pair_event = str(uuid.uuid4())
                st.session_state.answers = suggestion.get("answers", 0)
        if st.session_state.get("ended"):
            pass
        elif pending is not None:
            result = api("/api/comparisons", {
                "event_id": st.session_state.pending_pair_event,
                "session_id": sid,
                "item_a": pending["item_a"], "item_b": pending["item_b"],
            })
            st.session_state.pop("pending_pair", None)
            st.session_state.pop("pending_pair_event", None)
            result["answers"] = st.session_state.get("answers", 0)
            st.session_state.pair = result
        else:
            st.session_state.ended = True

    feedback = st.session_state.pop("choice_feedback", None)
    if feedback:
        st.markdown(
            f'<div class="choice-feedback"><span>✨</span><div>{html.escape(feedback)}</div></div>',
            unsafe_allow_html=True,
        )

    if not st.session_state.ended:
        pair = st.session_state.pair
        if not pair:
            st.info("Recherche d'une comparaison disponible…")
            return
        st.title("Laquelle choisirais-tu ?")
        answers = st.session_state.get("answers", pair.get("answers", 0))
        st.progress(min(answers / 10, 1.0), text=f"Choix {answers} sur 10")
        left, right = st.columns(2, gap="large")
        with left:
            with st.container(border=True):
                st.markdown('<p class="choice-index">Option 1</p>', unsafe_allow_html=True)
                item_card(pair["left"])
                if st.button("💚  Je préfère celle-ci", key="choose-left", type="primary", use_container_width=True):
                    submit_choice(pair, "left")
        with right:
            with st.container(border=True):
                st.markdown('<p class="choice-index">Option 2</p>', unsafe_allow_html=True)
                item_card(pair["right"])
                if st.button("💚  Je préfère celle-ci", key="choose-right", type="primary", use_container_width=True):
                    submit_choice(pair, "right")
        if st.button("Passer cette comparaison", key="skip-pair"):
            submit_choice(pair, "skip")
    else:
        st.title("Tes sorties recommandées")
        if st.session_state.get("stop_reason") == "no_informative_pair":
            answers = st.session_state.get("answers", 0)
            st.info(f"La comparaison s'est arrêtée après {answers} choix : il ne restait pas "
                    "de paire assez différente dans ce créneau. Le classement utilise les choix déjà faits.")
        elif st.session_state.get("stop_reason") == "max_offers":
            st.info("La séance a atteint sa limite de comparaisons. Le classement utilise les choix enregistrés.")
        recommendations = st.session_state.get("recommendations")
        if recommendations is None:
            result = api("/recommendations", {
                "event_id": event_id("recommendations-event"), "session_id": sid,
            })
            st.session_state.pop("recommendations-event", None)
            st.session_state.recommendations = result
            st.rerun()
            return
        items = recommendations.get("items", [])
        if not items:
            st.info("Aucune activité ne correspond à ces contraintes. Essaie un créneau, un budget ou un arrondissement plus large.")
            return
        st.caption(f"Classement personnalisé à partir de {st.session_state.get('answers', 0)} choix.")
        st.subheader("Meilleur match")
        for index, item in enumerate(items):
            if index == 1:
                st.subheader("Autres sorties recommandées")
            with st.container(border=True):
                if item.get("selected_by_user"):
                    st.success("Tu avais choisi cette sortie pendant les comparaisons.")
                item_card(item)
                c1, c2 = st.columns(2)
                with c1:
                    if item.get("source_url"):
                        if st.button("Ouvrir l'annonce", key=f"open-{item['impression_id']}", use_container_width=True):
                            api("/api/item_action", {"event_id": str(uuid.uuid4()), "session_id": sid,
                                 "impression_id": item["impression_id"], "action": "open"})
                        st.markdown(f"[Lien vers l'annonce]({item['source_url']})")
                with c2:
                    if st.button("Garder", key=f"save-{item['impression_id']}", use_container_width=True):
                        api("/api/item_action", {"event_id": str(uuid.uuid4()), "session_id": sid,
                             "impression_id": item["impression_id"], "action": "save"})
                        st.success("Enregistré pour cette démonstration.")
        show_map(items)


def submit_choice(pair: dict, choice: str) -> None:
    api("/log_interaction", {
        "event_id": event_id("interaction-event"),
        "session_id": st.session_state.session_id,
        "comparison_id": pair["comparison_id"], "choice": choice,
    })
    if choice == "skip":
        feedback = "On passe à une nouvelle comparaison."
    else:
        selected = pair["left"] if choice == "left" else pair["right"]
        feedback = f"Bien noté : {selected.get('title', 'ta sortie préférée')} — j’affine la suite."
    st.session_state.choice_feedback = feedback
    st.session_state.pop("interaction-event", None)
    st.session_state.pair = None
    st.rerun()


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


def render() -> None:
    st.set_page_config(page_title="SwiParis", page_icon="🗼", layout="wide")
    inject_theme()
    try:
        if not st.session_state.get("session_id"):
            start_screen()
        else:
            if st.sidebar.button("Nouvelle séance"):
                for key in ("session_id", "constraints", "pair", "ended", "recommendations",
                            "pending_pair", "pending_pair_event", "stop_reason", "answers",
                            "available_count"):
                    st.session_state.pop(key, None)
                st.rerun()
            pair_screen()
    except RuntimeError as exc:
        st.error(str(exc))


def main() -> None:
    """Launch Streamlit from the installed ``swiparis-streamlit`` command."""
    from streamlit.web import cli as stcli
    app_file = str(Path(__file__).resolve())
    sys.argv = ["streamlit", "run", app_file]
    stcli.main()


if __name__ == "__main__":
    render()

