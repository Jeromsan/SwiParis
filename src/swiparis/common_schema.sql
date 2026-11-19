PRAGMA foreign_keys=ON;
PRAGMA user_version=1;
CREATE TABLE metadata(key TEXT PRIMARY KEY,value_json TEXT NOT NULL);
CREATE TABLE categories(category TEXT PRIMARY KEY,label TEXT NOT NULL);
CREATE TABLE items(
 item_id TEXT PRIMARY KEY, source TEXT NOT NULL CHECK(source IN ('foursquare','paris_data')),
 source_id TEXT NOT NULL, title TEXT NOT NULL, description TEXT NOT NULL, lead_text TEXT NOT NULL,
 source_url TEXT, source_category_label TEXT, snapshot_id TEXT,
 price_type TEXT NOT NULL, is_free INTEGER, price_amount_eur REAL, price_detail TEXT NOT NULL,
 cover_url TEXT, cover_credit TEXT NOT NULL, booking_url TEXT,
 date_description TEXT NOT NULL, audience TEXT NOT NULL, quality_flags_json TEXT NOT NULL,
 UNIQUE(source,source_id)
);
CREATE TABLE item_categories(
 item_id TEXT NOT NULL REFERENCES items(item_id), category TEXT NOT NULL REFERENCES categories(category),
 PRIMARY KEY(item_id,category)
);
CREATE INDEX categories_items ON item_categories(category,item_id);
CREATE TABLE visits(
 user_id TEXT NOT NULL, source_user_id TEXT NOT NULL,
 item_id TEXT NOT NULL REFERENCES items(item_id), utc_time TEXT NOT NULL, timezone_offset INTEGER NOT NULL,
 PRIMARY KEY(user_id,utc_time,item_id)
) WITHOUT ROWID;
CREATE INDEX visits_item ON visits(item_id);
CREATE TABLE locations(
 item_id TEXT NOT NULL REFERENCES items(item_id),location_index INTEGER NOT NULL,
 venue_name TEXT,street TEXT,postal_code TEXT,city TEXT,arrondissement INTEGER,
 latitude REAL NOT NULL,longitude REAL NOT NULL,PRIMARY KEY(item_id,location_index)
);
CREATE TABLE occurrences(
 item_id TEXT NOT NULL REFERENCES items(item_id),start_utc TEXT NOT NULL,end_utc TEXT NOT NULL,
 source_start TEXT NOT NULL,source_end TEXT NOT NULL,
 PRIMARY KEY(item_id,start_utc,end_utc),CHECK(end_utc>start_utc)
);
CREATE INDEX occurrences_time ON occurrences(start_utc,end_utc);
CREATE VIEW training_visits AS
 SELECT v.*, c.category FROM visits v JOIN item_categories c USING(item_id);
CREATE VIEW demo_items AS SELECT * FROM items WHERE source='paris_data';
