-- ============================================================
-- What If Pipeline — Supabase pgvector Setup
-- Run this ONCE in your Supabase SQL Editor
-- (Database → SQL Editor → New Query → paste → Run)
--
-- Embedding model : all-MiniLM-L6-v2 (local, sentence-transformers)
-- Embedding dim   : 384
-- ============================================================

-- 1. Enable the pgvector extension (only needed once per project)
create extension if not exists vector;

-- 2. Create the ideas table
--    embedding dimension: 384 (all-MiniLM-L6-v2)
create table if not exists what_if_ideas (
    id          uuid        primary key default gen_random_uuid(),
    idea        text        not null,
    embedding   vector(384) not null,
    created_at  timestamptz not null default now()
);

-- 3. Disable Row Level Security
--    This is a private pipeline table (not user-facing), so RLS is not needed.
--    Without this, the anon key will be blocked from inserting rows.
alter table what_if_ideas disable row level security;

-- 3. Index for fast approximate nearest-neighbour search (cosine distance)
create index if not exists what_if_ideas_embedding_idx
    on what_if_ideas
    using ivfflat (embedding vector_cosine_ops)
    with (lists = 100);

-- 4. match_ideas RPC — called by the pipeline's similarity filter
--    Returns ideas ordered by cosine similarity (highest first).
--    similarity = 1.0 means identical, 0.0 means completely different.
create or replace function match_ideas(
    query_embedding  vector(384),
    match_threshold  float,
    match_count      int
)
returns table (
    id          uuid,
    idea        text,
    similarity  float
)
language sql stable
as $$
    select
        what_if_ideas.id,
        what_if_ideas.idea,
        1 - (what_if_ideas.embedding <=> query_embedding) as similarity
    from what_if_ideas
    where 1 - (what_if_ideas.embedding <=> query_embedding) >= match_threshold
    order by what_if_ideas.embedding <=> query_embedding
    limit match_count;
$$;

-- ============================================================
-- Verification: after setup, run this to confirm everything works:
--   select count(*) from what_if_ideas;
--   select * from match_ideas(array_fill(0::float, array[384])::vector, 0.0, 1);
-- ============================================================
