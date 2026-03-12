-- ============================================================
-- ABN Consulting AI Co-Navigator — Supabase Database Schema
-- Run this in your Supabase project's SQL Editor
-- ============================================================

-- Enable UUID extension (enabled by default in Supabase)
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ------------------------------------------------------------
-- Clients table
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS clients (
  id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  client_id    TEXT UNIQUE NOT NULL,
  name         TEXT NOT NULL,
  email        TEXT,
  created_at   TIMESTAMPTZ DEFAULT NOW()
);

-- ------------------------------------------------------------
-- Coaching sessions table
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS coaching_sessions (
  id                   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id           TEXT UNIQUE NOT NULL,
  client_id            TEXT NOT NULL REFERENCES clients(client_id) ON DELETE CASCADE,
  timestamp            TIMESTAMPTZ DEFAULT NOW(),
  focus_goal           TEXT,
  environmental_changes TEXT,
  mood_indicator       TEXT,
  alert_level          TEXT CHECK (alert_level IN ('green', 'yellow', 'red')),
  alert_reason         TEXT,
  summary_for_coach    TEXT,
  raw_conversation     JSONB
);

CREATE INDEX IF NOT EXISTS idx_sessions_client_id ON coaching_sessions(client_id);
CREATE INDEX IF NOT EXISTS idx_sessions_timestamp  ON coaching_sessions(timestamp DESC);

-- ------------------------------------------------------------
-- Key Results table (one row per KR per session)
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS key_results (
  id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id   TEXT NOT NULL REFERENCES coaching_sessions(session_id) ON DELETE CASCADE,
  kr_id        INTEGER NOT NULL,
  description  TEXT NOT NULL,
  status_pct   INTEGER CHECK (status_pct BETWEEN 0 AND 100),
  status_color TEXT CHECK (status_color IN ('green', 'yellow', 'red'))
);

CREATE INDEX IF NOT EXISTS idx_kr_session_id ON key_results(session_id);

-- ------------------------------------------------------------
-- Obstacles table (one row per obstacle per session)
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS obstacles (
  id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id   TEXT NOT NULL REFERENCES coaching_sessions(session_id) ON DELETE CASCADE,
  description  TEXT NOT NULL,
  reported_at  TIMESTAMPTZ DEFAULT NOW(),
  resolved     BOOLEAN DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_obstacles_session_id ON obstacles(session_id);

-- ------------------------------------------------------------
-- Row Level Security (RLS)
-- Only the service role (backend) can read/write.
-- The anon key used by Wix frontend cannot bypass this.
-- ------------------------------------------------------------
ALTER TABLE clients          ENABLE ROW LEVEL SECURITY;
ALTER TABLE coaching_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE key_results      ENABLE ROW LEVEL SECURITY;
ALTER TABLE obstacles        ENABLE ROW LEVEL SECURITY;

-- Drop existing policies if re-running
DROP POLICY IF EXISTS service_only ON clients;
DROP POLICY IF EXISTS service_only ON coaching_sessions;
DROP POLICY IF EXISTS service_only ON key_results;
DROP POLICY IF EXISTS service_only ON obstacles;

-- Service role has full access; all other roles are denied
CREATE POLICY service_only ON clients
  USING (auth.role() = 'service_role');

CREATE POLICY service_only ON coaching_sessions
  USING (auth.role() = 'service_role');

CREATE POLICY service_only ON key_results
  USING (auth.role() = 'service_role');

CREATE POLICY service_only ON obstacles
  USING (auth.role() = 'service_role');
