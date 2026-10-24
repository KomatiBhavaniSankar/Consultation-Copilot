-- Run once in Supabase: Dashboard -> SQL Editor -> New query -> paste -> Run.
-- (The app also runs the CREATE statements itself on start-up, so this is safe to run again.)

create table if not exists public.consultations (
    id              bigint generated always as identity primary key,
    created_at      text   not null,              -- ISO timestamp written by the app
    patient_name    text   not null,
    diagnosis       text,
    medicines       text,                         -- JSON list, stored as text
    vitals          text,
    doctor_approach text,
    transcript      text,
    dialogue        text,                         -- JSON list of {speaker, text}
    sheet           text,                         -- JSON take-home sheet
    prediction      text,                         -- JSON
    mode            text default 'lifestyle'      -- emergency | critical | lifestyle | maternity
);

create index if not exists consultations_mode_idx on public.consultations (mode);

-- SECURITY: the app talks to Postgres directly with the database password (server side only).
-- Turning on Row Level Security with NO policies, and revoking the API roles, means nobody can
-- read or write patient records through Supabase's public REST API / anon key.
alter table public.consultations enable row level security;
revoke all on public.consultations from anon, authenticated;
