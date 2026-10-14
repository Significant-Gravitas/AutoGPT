-- A hard forget marks with erasedAt every dream pass it may have reached
-- (backend/copilot/dream/forget_copies.py); the mark is never cleared. A pass
-- already closed has the dream's text in its row blanked by the forget itself,
-- and one still open has it blanked in the statement that closes it, whichever
-- way it closes (backend/data/dream_pass_update.py). A narrative written after
-- the mark is written blanked (backend/copilot/db.py, add_dream_narrative).
-- A nullable column with no default on an existing table: no row is rewritten.
-- Prisma does not track functions, so the two below cause no drift.

-- AlterTable
ALTER TABLE "DreamPass" ADD COLUMN "erasedAt" TIMESTAMP(3);

-- CreateFunction
-- value, with every string held under one of fields, at any depth, replaced
-- by placeholder. Anything else (numbers, uuids, nulls, nested objects under
-- those keys) stays.
CREATE OR REPLACE FUNCTION dream_text_blanked(value jsonb, fields text[], placeholder text)
RETURNS jsonb LANGUAGE plpgsql IMMUTABLE SET search_path FROM CURRENT AS $$
BEGIN
    IF jsonb_typeof(value) = 'object' THEN
        RETURN COALESCE((
            SELECT jsonb_object_agg(
                member.key,
                CASE WHEN member.key = ANY(fields) AND jsonb_typeof(member.item) = 'string'
                    THEN to_jsonb(placeholder)
                    ELSE dream_text_blanked(member.item, fields, placeholder) END)
            FROM jsonb_each(value) AS member(key, item)
        ), '{}'::jsonb);
    END IF;
    IF jsonb_typeof(value) = 'array' THEN
        RETURN COALESCE((
            SELECT jsonb_agg(
                dream_text_blanked(element.item, fields, placeholder)
                ORDER BY element.position)
            FROM jsonb_array_elements(value) WITH ORDINALITY AS element(item, position)
        ), '[]'::jsonb);
    END IF;
    RETURN value;
END;
$$;

-- CreateFunction
-- A pass's error with any text after its code replaced by placeholder. An
-- error that is a code alone ("<phase>: <code>", the form errors take since
-- this migration) stays; another keeps its leading "<phase>: " if it has one.
CREATE OR REPLACE FUNCTION dream_error_blanked(error text, placeholder text)
RETURNS text LANGUAGE sql IMMUTABLE AS $$
    SELECT CASE
        WHEN error IS NULL OR error ~ '^[a-z_]+: [A-Za-z0-9_]+$' THEN error
        WHEN error ~ '^[a-z_]+: ' THEN substring(error FROM '^[a-z_]+: ') || placeholder
        ELSE placeholder
    END
$$;
