"""Streamlit front end for the local SwiParis FastAPI demo."""
from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
import html
import json
import os
from pathlib import Path
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import uuid
from zoneinfo import ZoneInfo

import streamlit as st
import streamlit.components.v1 as components


API_URL = os.environ.get("SWIPARIS_API_URL", "http://127.0.0.1:8000")


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


def start_screen() -> None:
    st.title("SwiParis")
    st.caption("Choisis une sortie parmi deux, puis découvre une sélection adaptée à tes goûts.")
    health = api("/health")
    catalog = api("/api/catalog?limit=1")
    category_labels = catalog.get("categories", {})

    with st.form("session-setup"):
        st.subheader("Quand et où ?")
        day = st.date_input("Date de sortie", value=date.today())
        start_time = st.time_input("Début du créneau", value=time(14, 0), step=900)
        duration = st.selectbox("Durée", (60, 90, 120, 180), format_func=lambda n: f"{n} minutes")
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
    start_utc, end_utc = paris_iso(day, start_time, int(duration))
    constraints = {"start_utc": start_utc, "end_utc": end_utc}
    if arrondissement is not None:
        constraints["arrondissement"] = arrondissement
    if category:
        constraints["category"] = category
    if use_budget:
        constraints["budget_eur"] = float(budget)
    created = api("/api/sessions", {
        "event_id": event_id("create-session-event"), "consent": True,
        "constraints": constraints, "catalog_version": health["catalog_version"],
        "model_version": health["model_version"], "data_origin": "pilot",
    })
    st.session_state.session_id = created["session_id"]
    st.session_state.constraints = constraints
    st.session_state.pair = None
    st.session_state.ended = False
    st.session_state.recommendations = None
    st.rerun()


def item_card(item: dict) -> None:
    st.markdown(f"### {item['title']}")
    st.caption(" · ".join(item.get("categories", [])))
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
                pending = None
            else:
                pending = suggestion["comparison"]
                st.session_state.pending_pair = pending
                st.session_state.pending_pair_event = str(uuid.uuid4())
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
            st.session_state.pair = result
        else:
            st.session_state.ended = True

    if not st.session_state.ended:
        pair = st.session_state.pair
        if not pair:
            st.info("Recherche d'une comparaison disponible…")
            return
        st.title("Laquelle choisirais-tu ?")
        st.progress(min(pair.get("answers", 0) / 5, 1.0), text=f"Choix {pair.get('answers', 0)} sur 5")
        left, right = st.columns(2, gap="large")
        with left:
            with st.container(border=True):
                item_card(pair["left"])
                if st.button("Je choisis cette sortie", key="choose-left", type="primary", use_container_width=True):
                    submit_choice(pair, "left")
        with right:
            with st.container(border=True):
                item_card(pair["right"])
                if st.button("Je choisis cette sortie", key="choose-right", type="primary", use_container_width=True):
                    submit_choice(pair, "right")
        if st.button("Passer cette comparaison", key="skip-pair"):
            submit_choice(pair, "skip")
    else:
        st.title("Tes sorties recommandées")
        recommendations = st.session_state.get("recommendations")
        if recommendations is None:
            if st.button("Afficher les recommandations", type="primary"):
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
        for item in items:
            with st.container(border=True):
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
    st.session_state.pop("interaction-event", None)
    st.session_state.pair = None
    st.rerun()


def show_map(items: list[dict]) -> None:
    config = api("/api/map-config")
    key = config.get("api_key", "")
    markers = []
    for item in items:
        for location in item.get("locations", [])[:1]:
            if location.get("latitude") is not None and location.get("longitude") is not None:
                markers.append({"lat": location["latitude"], "lng": location["longitude"],
                                "title": item["title"], "url": item.get("source_url", "")})
    if not markers:
        return
    st.subheader("Carte des recommandations")
    if not key:
        st.caption("La carte Google est désactivée. Les annonces et leurs liens restent disponibles.")
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
</script><script async src="https://maps.googleapis.com/maps/api/js?key={html.escape(key, quote=True)}&callback=initMap"></script>
</body></html>"""
    components.html(html_doc, height=450, scrolling=False)


def render() -> None:
    st.set_page_config(page_title="SwiParis", page_icon="🗼", layout="wide")
    try:
        if not st.session_state.get("session_id"):
            start_screen()
        else:
            if st.sidebar.button("Nouvelle séance"):
                for key in ("session_id", "constraints", "pair", "ended", "recommendations"):
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

