"""Même parcours (5 paires, mêmes réponses), historiques différents : listes différentes.

    python scripts/demo_personas.py
"""
import json, numpy as np, sys
sys.path.insert(0, "src")  # lancer depuis la racine du dépôt
from swiparis.catalog import Catalog
from swiparis.ml.items import encode_common
from swiparis.ml.personas import PersonaModel, PersonaProfile, habit_vector, known_history, visits_with_buckets
from swiparis.ml.selection import next_pair, rank
from swiparis.ml.splits import load_splits
from swiparis.ml.model import SessionProfile
from swiparis.ml.prior import load_prior

db = "data/processed/common.sqlite"
splits = load_splits("data/processed/splits.csv")
train = {u for u, s in splits.items() if s == "train"}
fsq = encode_common(db, users=train, sources=["foursquare"], popularity_users=train)
paris = encode_common(db, users=train, sources=["paris_data"], popularity_users=train)
personas = PersonaModel.load("models/personas.json", paris.space)
prior = load_prior("models/prior_w0_phi.json", paris.space)
rows = Catalog(db).list_items({"start_utc": "2026-11-28T13:00:00Z", "end_utc": "2026-11-28T16:00:00Z"}, limit=None)["items"]
title = {r["item_id"]: r["title"] for r in rows}
ids = [r["item_id"] for r in rows if r["item_id"] in set(paris.ids)]
phi = paris.rows(ids)
# même parcours pour tout le monde : 5 paires fixées, réponse « gauche » à chaque fois
guide, rng, pairs = SessionProfile(prior, eta=0.05), np.random.default_rng(7), []
for _ in range(5):
    a, b = next_pair(guide, phi, ids, pairs, rng=rng)
    pairs.append((a, b)); guide.answer(phi[ids.index(a)], phi[ids.index(b)], "left")
test_users = sorted(u for u, s in splits.items() if s == "test")
visits = visits_with_buckets(db, test_users)
habits = {u: habit_vector(known_history(visits[u]), fsq) for u in test_users}
by_persona = {}
for u in test_users:
    by_persona.setdefault(int(np.argmax(personas.start(habits[u]))), u)
ua, ub = by_persona[0], by_persona[4]   # deux historiques de personas différentes
def run(start, label):
    profile = PersonaProfile(personas, start=start, eta=0.05)
    for a, b in pairs:
        profile.answer(phi[ids.index(a)], phi[ids.index(b)], "left")
    top = [i for i, _ in rank(profile, phi, ids, k=10, diversity=0.3)]
    pi = profile.weights_pi
    main = np.argsort(-pi)[:2]
    print(f"\n{label}\n  ressemble à : " + " ; ".join(f"{personas.names[j]} ({pi[j]:.0%})" for j in main))
    for n, i in enumerate(top[:5], 1):
        print(f"  {n}. {title[i][:70]}")
    return top
print("Créneau : samedi 28/11, 14 h – 17 h ;", len(ids), "activités. Mêmes 5 paires, mêmes réponses pour les trois.")
t0 = run(None, "Sans historique (nouvel utilisateur)")
ta = run(personas.start(habits[ua]), f"Historique de {ua} (persona dominante : {personas.names[0]})")
tb = run(personas.start(habits[ub]), f"Historique de {ub} (persona dominante : {personas.names[4]})")
print(f"\nActivités communes dans le top 10 : A et B {len(set(ta) & set(tb))}/10 ; sans historique et A {len(set(t0) & set(ta))}/10 ; sans historique et B {len(set(t0) & set(tb))}/10")
