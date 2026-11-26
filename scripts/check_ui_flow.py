"""Recette de l'interface Streamlit, comme un utilisateur : parcours complet, garder plusieurs
sorties, retirer, annuler, exporter, revenir plus tard avec la même adresse.

L'API doit tourner, avec la vraie base (le scénario utilise le samedi 28/11, toute la journée) :

    .venv/bin/python -m uvicorn swiparis.api:app --host 127.0.0.1 --port 8000      # fenêtre 1
    .venv/bin/python scripts/check_ui_flow.py                                      # fenêtre 2

SWIPARIS_API_URL (défaut http://127.0.0.1:8000) et SWIPARIS_LOGS (défaut
data/pilot/interactions.sqlite) doivent désigner la même API et sa base de logs.
Les séances créées sont des séances de test enregistrées dans cette base.
"""
import datetime, json, os, sqlite3, sys, urllib.error, urllib.request
from pathlib import Path
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
SRC = str(ROOT / "src")
os.environ.setdefault("SWIPARIS_API_URL", "http://127.0.0.1:8000")
SCRIPT = f"import sys; sys.path.insert(0, '{SRC}')\nfrom swiparis.streamlit_app import render\nrender()"
sys.path.insert(0, SRC)
LOGS = os.environ.get("SWIPARIS_LOGS", str(ROOT / "data/pilot/interactions.sqlite"))
results = []

def check(label, ok, detail=""):
    ok = bool(ok); results.append(ok)
    print(("OK   " if ok else "ÉCHEC"), label, ("— " + str(detail)) if detail else "")

def no_errors(at, step):
    errs = [e.value for e in at.error] + [str(x.value)[:200] for x in at.exception]
    check(f"aucune erreur affichée ({step})", not errs, errs)

def btn(at, label=None, key=None, startswith=None):
    for b in at.button:
        if (key and b.key == key) or (label and b.label == label) or (startswith and b.label.startswith(startswith)):
            return b
    raise AssertionError(f"bouton introuvable: {label or key or startswith}; présents: {[x.label for x in at.button]}")

def nav_label(at):
    return at.radio[0].options

def count_in_nav(at):
    import re
    return int(re.search(r"\((\d+)\)", [o for o in at.radio[0].options if "sélection" in o][0]).group(1))

at = AppTest.from_string(SCRIPT, default_timeout=60).run()
no_errors(at, "accueil")
carnet = at.query_params.get("carnet")
carnet = carnet[0] if isinstance(carnet, list) else carnet
check("un identifiant de sélection est mis dans l'adresse", bool(carnet) and len(carnet) >= 8, carnet)
d = at.date_input[0]
check("jour borné à la liste d'activités", d.proto.min and d.proto.max, (d.proto.min, d.proto.max))
check("disponibilité : toute la journée par défaut", [s.value for s in at.selectbox if s.label == "Disponibilité"] == ["Toute la journée"])
btn(at, "Commencer").click().run()
check("sans accord : message clair et pas de démarrage", any("Coche la case" in e.value for e in at.error))
for c in at.checkbox:
    if "anonyme" in c.label:
        c.check()
at.date_input[0].set_value(datetime.date(2026, 11, 28))
btn(at, "Commencer").click().run()
no_errors(at, "démarrage")
check("écran des choix affiché", any(t.value == "Laquelle choisirais-tu ?" for t in at.title))
check("compteur de sélection à 0", count_in_nav(at) == 0, nav_label(at))

# Garder pendant les choix
btn(at, key="pairkeep-left").click().run()
no_errors(at, "garder pendant un choix")
check("garder pendant un choix : compteur à 1", count_in_nav(at) == 1, nav_label(at))
check("garder pendant un choix : message de confirmation", any("1 sortie ajoutée" in s.value for s in at.success),
      [s.value for s in at.success])
check("le bouton montre l'état réel", btn(at, key="pairkeep-left").label.startswith("★ Dans ta sélection"))
pair_kept = btn(at, key="pairkeep-left")

progress = []
for _ in range(10):
    progress += [p.proto.text for p in at.get("progress")]
    if any(b.key == "choose-right" for b in at.button):
        btn(at, key="choose-right").click().run()
    else:
        break
no_errors(at, "fin des choix")
check("progression 0 → 9 sur 10 affichée", progress[:10] == [f"Choix {i} sur 10" for i in range(10)], progress)
checkboxes = [c for c in at.checkbox if (c.key or "").startswith("pick-reco-")]
check("3 recommandations affichées sans clic en plus, avec une case à cocher", len(checkboxes) == 3, len(checkboxes))
check("persona affichée avec son pourcentage", any("Tes choix ressemblent le plus à" in m.value for m in at.markdown))
links = [e.proto.url for e in at.get("link_button") if "/go/" in e.proto.url]
check("3 liens « Voir l'annonce » qui passent par /go", len(links) == 3, links[:1])
# Le lien ouvre bien l'annonce (redirection) et l'ouverture est journalisée
class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k): return None
try:
    urllib.request.build_opener(NoRedirect).open(links[0])
    status, location = 200, None
except urllib.error.HTTPError as exc:
    status, location = exc.code, exc.headers.get("location")
check("le lien redirige vers paris.fr", status == 303 and (location or "").startswith("https://"), (status, location))

garder = btn(at, startswith="☆ Garder la")
check("bouton de lot désactivé tant que rien n'est coché", garder.disabled)
for c in checkboxes[:2]:
    c.check()
at.run()
lot = btn(at, startswith="☆ Garder les 2")
check("bouton de lot : « Garder les 2 sorties cochées »", not lot.disabled, lot.label)
lot.click().run()
no_errors(at, "garder en lot")
check("lot : message « 2 sorties ajoutées »", any("2 sorties ajoutées" in s.value for s in at.success),
      [s.value for s in at.success])
check("lot : compteur à 3", count_in_nav(at) == 3, nav_label(at))
check("lot : cases décochées après l'action", not any(c.value for c in at.checkbox if (c.key or "").startswith("pick-reco-")))
check("les sorties gardées n'ont plus de case à cocher", len([c for c in at.checkbox if (c.key or "").startswith("pick-reco-")]) == 1)
# garder une seule, puis la retirer depuis la liste
single = [b for b in at.button if (b.key or "").startswith("keep-") and b.label == "☆ Garder"][0]
single_key = single.key
single.click().run()
check("garder une seule sortie : compteur à 4", count_in_nav(at) == 4, nav_label(at))
btn(at, key=single_key).click().run()
check("retirer depuis les recommandations : compteur à 3", count_in_nav(at) == 3, nav_label(at))
check("carte affichée sans clé Google", any(getattr(e.proto, "html", None) or e.type == "iframe" for e in at.get("iframe"))
      or len(at.get("iframe")) >= 1 or any("Carte des 3 recommandations" in h.value for h in at.subheader))

# Page Ma sélection
btn(at, startswith="Voir ma sélection").click().run()
no_errors(at, "ma sélection")
check("page Ma sélection ouverte", any(t.value == "Ma sélection" for t in at.title))
sel = [c for c in at.checkbox if (c.key or "").startswith("pick-sel-")]
check("3 sorties listées", len(sel) == 3, len(sel))
downloads = at.get("download_button")
check("exports agenda et tableau proposés", len(downloads) == 2, [d.proto.label for d in downloads])
sel[0].check(); sel[1].check(); at.run()
btn(at, startswith="Retirer (2)").click().run()
check("retrait en lot : demande de confirmation", any("Retirer 2 sorties" in w.value for w in at.warning))
btn(at, "Oui, retirer").click().run()
no_errors(at, "retrait en lot")
check("retrait en lot : compteur à 1", count_in_nav(at) == 1, nav_label(at))
check("retrait en lot : proposition d'annuler", any("2 sorties retirées" in i.value for i in at.info))
btn(at, "Annuler").click().run()
check("annuler : compteur revenu à 3", count_in_nav(at) == 3, nav_label(at))
check("annuler : message de confirmation", any("2 sorties remises" in s.value for s in at.success))
[b for b in at.button if (b.key or "").startswith("remove-")][0].click().run()
check("retirer une seule : compteur à 2", count_in_nav(at) == 2, nav_label(at))

# Exports : contenu réel
from swiparis.streamlit_app import ics_text, csv_text
saved = json.load(urllib.request.urlopen(f"{os.environ['SWIPARIS_API_URL']}/api/saved?owner_id={carnet}"))["items"]
ics = ics_text(saved)
check("agenda : 2 événements, format iCalendar", ics.count("BEGIN:VEVENT") == 2 and ics.startswith("BEGIN:VCALENDAR\r\n")
      and all(len(l.encode()) <= 75 for l in ics.split("\r\n")), ics.count("BEGIN:VEVENT"))
starts = [l for l in ics.split("\r\n") if l.startswith("DTSTART:")]
check("agenda : horaires le samedi 28/11 (journée à Paris)", all(l[8:16] == "20261128" for l in starts), starts)
table = csv_text(saved).splitlines()
check("tableau : en-tête + 2 lignes", len(table) == 3 and table[0].startswith("Titre;Quand;Lieu"), table[0])

# Persistance : nouvelle visite avec la même adresse
again = AppTest.from_string(SCRIPT, default_timeout=60)
again.query_params["carnet"] = carnet
again.run()
again.radio[0].set_value("selection").run()
check("nouvelle visite, même adresse : les 2 sorties sont là",
      len([c for c in again.checkbox if (c.key or "").startswith("pick-sel-")]) == 2)
other = AppTest.from_string(SCRIPT, default_timeout=60).run()
other.radio[0].set_value("selection").run()
check("autre adresse : sélection vide et message d'aide", any("vide" in i.value for i in other.info))
btn(other, "Trouver une sortie").click().run()
check("état vide : le bouton ramène à la recherche", any(b.label == "Commencer" for b in other.button))

# Base de données
session_id = at.session_state["session_id"]
with sqlite3.connect(LOGS) as db:
    rows = db.execute("SELECT count(*), sum(removed_at_utc IS NULL) FROM saved_items WHERE owner_id=?", (carnet,)).fetchone()
    actions = ("SELECT count(*) FROM recommendation_actions a JOIN recommendations_shown r USING(impression_id) "
               "WHERE r.session_id=? AND a.action=?")
    saves = db.execute(actions, (session_id, "save")).fetchone()[0]
    opens = db.execute(actions, (session_id, "open")).fetchone()[0]
check("base : 4 lignes pour ce carnet dont 2 actives", rows == (4, 2), rows)
check("base : signal « save » pour l'évaluation (3 recos gardées)", saves == 3, saves)
check("base : ouverture de l'annonce journalisée", opens == 1, opens)
print(f"\n{sum(results)}/{len(results)} vérifications OK")
sys.exit(0 if all(results) else 1)
