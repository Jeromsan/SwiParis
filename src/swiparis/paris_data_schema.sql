PRAGMA foreign_keys=ON;
PRAGMA user_version=1;
CREATE TABLE snapshots (
    snapshot_id TEXT PRIMARY KEY,
    schema_version INTEGER NOT NULL,
    source_url TEXT NOT NULL,
    source_sha256 TEXT NOT NULL,
    as_of_utc TEXT NOT NULL,
    window_end_utc TEXT NOT NULL,
    manifest_json TEXT NOT NULL
);
CREATE TABLE events (
    event_id TEXT PRIMARY KEY,
    snapshot_id TEXT NOT NULL REFERENCES snapshots(snapshot_id),
    source_id TEXT NOT NULL UNIQUE,
    source_event_id TEXT,
    source_url TEXT NOT NULL,
    title TEXT NOT NULL,
    lead_text TEXT NOT NULL,
    description TEXT NOT NULL,
    price_type TEXT NOT NULL CHECK(price_type IN ('free','conditional_free','paid','unknown')),
    is_free INTEGER CHECK(is_free IN (0,1)),
    price_amount_eur REAL,
    price_detail TEXT NOT NULL,
    date_description TEXT NOT NULL,
    source_date_start TEXT,
    source_date_end TEXT,
    schedule_source TEXT NOT NULL CHECK(schedule_source IN ('occurrences','single_interval')),
    updated_at_utc TEXT,
    cover_url TEXT,
    cover_credit TEXT NOT NULL,
    booking_url TEXT,
    access_type TEXT NOT NULL,
    audience TEXT NOT NULL,
    quality_flags_json TEXT NOT NULL
);
CREATE TABLE event_locations (
    event_id TEXT NOT NULL REFERENCES events(event_id),
    location_index INTEGER NOT NULL,
    venue_name TEXT NOT NULL,
    street TEXT NOT NULL,
    postal_code TEXT NOT NULL,
    city TEXT NOT NULL,
    arrondissement INTEGER CHECK(arrondissement BETWEEN 1 AND 20),
    latitude REAL NOT NULL,
    longitude REAL NOT NULL,
    PRIMARY KEY(event_id,location_index)
);
CREATE TABLE event_occurrences (
    event_id TEXT NOT NULL REFERENCES events(event_id),
    start_utc TEXT NOT NULL,
    end_utc TEXT NOT NULL CHECK(end_utc > start_utc),
    source_start TEXT NOT NULL,
    source_end TEXT NOT NULL,
    PRIMARY KEY(event_id,start_utc,end_utc)
);
CREATE INDEX occurrences_time ON event_occurrences(start_utc,end_utc);
CREATE TABLE event_categories (
    event_id TEXT NOT NULL REFERENCES events(event_id),
    category TEXT NOT NULL,
    PRIMARY KEY(event_id,category)
);
CREATE TABLE event_tags (
    event_id TEXT NOT NULL REFERENCES events(event_id),
    source_tag TEXT NOT NULL,
    PRIMARY KEY(event_id,source_tag)
);
CREATE TABLE import_rejections (
    source_row INTEGER PRIMARY KEY,
    source_id TEXT,
    reason TEXT NOT NULL,
    kept_source_id TEXT
);
