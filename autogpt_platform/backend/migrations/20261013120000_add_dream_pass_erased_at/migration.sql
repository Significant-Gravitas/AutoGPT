-- A hard forget marks with erasedAt every dream pass it reached
-- (backend/copilot/dream/forget_copies.py); the mark is never cleared. A pass
-- is reached when it read a fact or episode the forget reached (readUuids:
-- every one it gathered, recorded before its first model call and never
-- cleared), may still be writing (open, or closed with its cleanup pending),
-- or wrote what the forget erased. A pass already closed has the dream's text
-- in its row blanked by the forget itself, and one still open has it blanked
-- in the statement that closes it, whichever way it closes
-- (backend/data/dream_pass_update.py). A narrative written after the mark is
-- written blanked, and carries its pass's readUuids into its session's
-- metadata, where a forget finds it once the pass's row has aged out
-- (backend/copilot/db.py, add_dream_narrative).
-- A nullable column and an array with a constant default on an existing
-- table: no row is rewritten.
-- Prisma does not track functions, so the two below cause no drift.

-- AlterTable
ALTER TABLE "DreamPass" ADD COLUMN     "erasedAt" TIMESTAMP(3),
ADD COLUMN     "readUuids" TEXT[] DEFAULT ARRAY[]::TEXT[];

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
-- A pass's error when it is "<stage>: <code>" with the stage one of stages
-- and the code one of codes (backend/copilot/dream/pass_errors.py, the only
-- errors a pass records). Any other is placeholder, after its "<stage>: "
-- when that stage is one of stages.
CREATE OR REPLACE FUNCTION dream_error_blanked(
    error text, stages text[], codes text[], placeholder text)
RETURNS text LANGUAGE sql IMMUTABLE AS $$
    SELECT CASE
        WHEN error IS NULL THEN NULL
        WHEN position(': ' IN error) = 0
            OR NOT (split_part(error, ': ', 1) = ANY(stages))
        THEN placeholder
        WHEN substr(error, position(': ' IN error) + 2) = ANY(codes) THEN error
        ELSE split_part(error, ': ', 1) || ': ' || placeholder
    END
$$;
