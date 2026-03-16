-- Run this in your Supabase SQL editor to create the bookings table.
-- Dashboard: https://app.supabase.com → your project → SQL Editor

create table if not exists public.bookings (
  id            uuid        primary key default gen_random_uuid(),
  name          text        not null,
  email         text        not null,
  subject       text,
  start_time    timestamptz not null,
  end_time      timestamptz not null,
  duration_mins integer     not null,
  meet_link     text,
  event_id      text,
  user_tz       text,
  created_at    timestamptz default now()
);

-- Index for querying by date range (admin dashboard use)
create index if not exists bookings_start_time_idx on public.bookings(start_time);

-- Index for looking up by email
create index if not exists bookings_email_idx on public.bookings(email);

-- Row Level Security: allow insert from anon (the React app uses the anon key)
alter table public.bookings enable row level security;

create policy "Allow anonymous insert"
  on public.bookings for insert
  to anon
  with check (true);

-- Only authenticated users (e.g. admin) can read bookings
create policy "Allow authenticated read"
  on public.bookings for select
  to authenticated
  using (true);
