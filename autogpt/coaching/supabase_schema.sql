-- ============================================================
-- ABN Consulting AI Co-Navigator — Supabase Database Schema
-- Run this in your Supabase project's SQL Editor
-- ============================================================

CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ============================================================
-- 1. USER PROFILES
-- Custom user table (email/password or Google OAuth).
-- ============================================================
CREATE TABLE IF NOT EXISTS user_profiles (
  user_id       UUID    PRIMARY KEY DEFAULT gen_random_uuid(),
  name          TEXT    NOT NULL,
  email         TEXT    NOT NULL UNIQUE,
  password_hash TEXT,                      -- NULL for Google-only accounts
  google_id     TEXT    UNIQUE,            -- NULL for email/password accounts
  created_at    TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_user_email    ON user_profiles(email);
CREATE INDEX IF NOT EXISTS idx_user_google   ON user_profiles(google_id);

-- ============================================================
-- 2. OBJECTIVES  (user's ongoing OKR plan)
-- ============================================================
CREATE TABLE IF NOT EXISTS objectives (
  objective_id  UUID    PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id       UUID    NOT NULL REFERENCES user_profiles(user_id) ON DELETE CASCADE,
  title         TEXT    NOT NULL,
  description   TEXT    NOT NULL DEFAULT '',
  status        TEXT    NOT NULL DEFAULT 'active'
                        CHECK (status IN ('active', 'archived', 'on_hold')),
  created_at    TIMESTAMPTZ DEFAULT NOW(),
  updated_at    TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_objectives_user ON objectives(user_id, status);

-- ============================================================
-- 3. USER KEY RESULTS  (master KR list, per objective)
-- ============================================================
CREATE TABLE IF NOT EXISTS user_key_results (
  kr_id         UUID    PRIMARY KEY DEFAULT gen_random_uuid(),
  objective_id  UUID    NOT NULL REFERENCES objectives(objective_id) ON DELETE CASCADE,
  user_id       UUID    NOT NULL REFERENCES user_profiles(user_id) ON DELETE CASCADE,
  description   TEXT    NOT NULL,
  status        TEXT    NOT NULL DEFAULT 'active'
                        CHECK (status IN ('active', 'archived', 'on_hold')),
  current_pct   INTEGER NOT NULL DEFAULT 0 CHECK (current_pct BETWEEN 0 AND 100),
  created_at    TIMESTAMPTZ DEFAULT NOW(),
  updated_at    TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_ukr_objective ON user_key_results(objective_id);
CREATE INDEX IF NOT EXISTS idx_ukr_user      ON user_key_results(user_id, status);

-- ============================================================
-- 4. CLIENTS  (legacy identifier, kept for dashboard compatibility)
-- ============================================================
CREATE TABLE IF NOT EXISTS clients (
  id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  client_id  TEXT UNIQUE NOT NULL,
  name       TEXT NOT NULL,
  email      TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================
-- 5. COACHING SESSIONS
-- ============================================================
CREATE TABLE IF NOT EXISTS coaching_sessions (
  id                    UUID    PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id            TEXT    UNIQUE NOT NULL,
  client_id             TEXT    NOT NULL REFERENCES clients(client_id) ON DELETE CASCADE,
  user_id               UUID    REFERENCES user_profiles(user_id) ON DELETE SET NULL,
  timestamp             TIMESTAMPTZ DEFAULT NOW(),
  focus_goal            TEXT,
  environmental_changes TEXT,
  mood_indicator        TEXT,
  alert_level           TEXT    CHECK (alert_level IN ('green', 'yellow', 'red')),
  alert_reason          TEXT,
  summary_for_coach     TEXT,
  raw_conversation      JSONB
);

CREATE INDEX IF NOT EXISTS idx_sessions_client_id ON coaching_sessions(client_id);
CREATE INDEX IF NOT EXISTS idx_sessions_user_id   ON coaching_sessions(user_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_sessions_timestamp  ON coaching_sessions(timestamp DESC);

-- ============================================================
-- 6. SESSION KEY RESULTS  (snapshot per session, not master plan)
-- ============================================================
CREATE TABLE IF NOT EXISTS key_results (
  id           UUID    PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id   TEXT    NOT NULL REFERENCES coaching_sessions(session_id) ON DELETE CASCADE,
  kr_id        INTEGER NOT NULL,
  description  TEXT    NOT NULL,
  status_pct   INTEGER CHECK (status_pct BETWEEN 0 AND 100),
  status_color TEXT    CHECK (status_color IN ('green', 'yellow', 'red'))
);

CREATE INDEX IF NOT EXISTS idx_kr_session_id ON key_results(session_id);

-- ============================================================
-- 7. SESSION OBSTACLES
-- ============================================================
CREATE TABLE IF NOT EXISTS obstacles (
  id           UUID    PRIMARY KEY DEFAULT gen_random_uuid(),
  session_id   TEXT    NOT NULL REFERENCES coaching_sessions(session_id) ON DELETE CASCADE,
  description  TEXT    NOT NULL,
  reported_at  TIMESTAMPTZ DEFAULT NOW(),
  resolved     BOOLEAN DEFAULT FALSE
);

CREATE INDEX IF NOT EXISTS idx_obstacles_session_id ON obstacles(session_id);

-- ============================================================
-- Row Level Security — service role only
-- ============================================================
ALTER TABLE user_profiles     ENABLE ROW LEVEL SECURITY;
ALTER TABLE objectives         ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_key_results   ENABLE ROW LEVEL SECURITY;
ALTER TABLE clients            ENABLE ROW LEVEL SECURITY;
ALTER TABLE coaching_sessions  ENABLE ROW LEVEL SECURITY;
ALTER TABLE key_results        ENABLE ROW LEVEL SECURITY;
ALTER TABLE obstacles          ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS service_only ON user_profiles;
DROP POLICY IF EXISTS service_only ON objectives;
DROP POLICY IF EXISTS service_only ON user_key_results;
DROP POLICY IF EXISTS service_only ON clients;
DROP POLICY IF EXISTS service_only ON coaching_sessions;
DROP POLICY IF EXISTS service_only ON key_results;
DROP POLICY IF EXISTS service_only ON obstacles;

CREATE POLICY service_only ON user_profiles    USING (auth.role() = 'service_role');
CREATE POLICY service_only ON objectives        USING (auth.role() = 'service_role');
CREATE POLICY service_only ON user_key_results  USING (auth.role() = 'service_role');
CREATE POLICY service_only ON clients           USING (auth.role() = 'service_role');
CREATE POLICY service_only ON coaching_sessions USING (auth.role() = 'service_role');
CREATE POLICY service_only ON key_results       USING (auth.role() = 'service_role');
CREATE POLICY service_only ON obstacles         USING (auth.role() = 'service_role');
