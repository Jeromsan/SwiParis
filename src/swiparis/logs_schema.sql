PRAGMA foreign_keys=ON;
PRAGMA user_version=2;
CREATE TABLE IF NOT EXISTS sessions(
 session_id TEXT PRIMARY KEY,created_at_utc TEXT NOT NULL,constraints_json TEXT NOT NULL,
 catalog_version TEXT NOT NULL,model_version TEXT NOT NULL,consent INTEGER NOT NULL CHECK(consent=1),
 data_origin TEXT NOT NULL CHECK(data_origin IN ('pilot','test'))
);
CREATE TABLE IF NOT EXISTS comparisons(
 comparison_id TEXT PRIMARY KEY,session_id TEXT NOT NULL REFERENCES sessions(session_id),
 left_item_id TEXT NOT NULL,right_item_id TEXT NOT NULL,offered_at_utc TEXT NOT NULL,
 choice TEXT CHECK(choice IN ('left','right','skip')),answered_at_utc TEXT,
 answer_event_id TEXT UNIQUE,
 CHECK(left_item_id<>right_item_id),CHECK((choice IS NULL)=(answered_at_utc IS NULL))
);
CREATE INDEX IF NOT EXISTS comparisons_session ON comparisons(session_id,offered_at_utc);
CREATE TABLE IF NOT EXISTS recommendations_shown(
 impression_id TEXT PRIMARY KEY,batch_event_id TEXT NOT NULL,
 session_id TEXT NOT NULL REFERENCES sessions(session_id),item_id TEXT NOT NULL,
 rank INTEGER NOT NULL CHECK(rank BETWEEN 1 AND 10),shown_at_utc TEXT NOT NULL,
 UNIQUE(batch_event_id,rank),UNIQUE(batch_event_id,item_id)
);
CREATE TABLE IF NOT EXISTS recommendation_actions(
 action_event_id TEXT PRIMARY KEY,impression_id TEXT NOT NULL REFERENCES recommendations_shown(impression_id),
 action TEXT NOT NULL CHECK(action IN ('open','save')),at_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS request_events(
 event_id TEXT PRIMARY KEY,operation TEXT NOT NULL,payload_sha256 TEXT NOT NULL,
 result_json TEXT NOT NULL,at_utc TEXT NOT NULL
);
-- Sélection de l'utilisateur (« Ma sélection »). État courant, pas un journal :
-- une ligne par (carnet, activité) ; retirer = removed_at_utc rempli, re-garder = remis à NULL.
-- owner_id est un identifiant aléatoire porté par l'adresse de la page, sans donnée personnelle.
CREATE TABLE IF NOT EXISTS saved_items(
 owner_id TEXT NOT NULL,item_id TEXT NOT NULL,
 session_id TEXT NOT NULL REFERENCES sessions(session_id),
 impression_id TEXT REFERENCES recommendations_shown(impression_id),
 saved_at_utc TEXT NOT NULL,removed_at_utc TEXT,
 PRIMARY KEY(owner_id,item_id)
);
CREATE INDEX IF NOT EXISTS saved_items_owner ON saved_items(owner_id,removed_at_utc,saved_at_utc);
