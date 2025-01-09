/*
  Warnings:
  - You are about replace a single brace string input format for the following blocks:
        - AgentOutputBlock
        - FillTextTemplateBlock
        - AITextGeneratorBlock
        - AIStructuredResponseGeneratorBlock
    with a double brace format.
   - This migration can be slow for a large updated AgentNode tables.
*/
BEGIN;
SET LOCAL statement_timeout = '10min';

WITH to_update AS (
    SELECT
        "id",
        "agentBlockId",
        "constantInput"::jsonb AS j
    FROM "AgentNode"
    WHERE
        "agentBlockId" IN (
            '363ae599-353e-4804-937e-b2ee3cef3da4', -- AgentOutputBlock
            'db7d8f02-2f44-4c55-ab7a-eae0941f0c30', -- FillTextTemplateBlock
            '1f292d4a-41a4-4977-9684-7c8d560b9f91', -- AITextGeneratorBlock
            'ed55ac19-356e-4243-a6cb-bc599e9b716f' -- AIStructuredResponseGeneratorBlock
        )
        AND (
             "constantInput"::jsonb->>'format'    ~ '(?<!\{)\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}(?!\})'
          OR "constantInput"::jsonb->>'prompt'    ~ '(?<!\{)\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}(?!\})'
          OR "constantInput"::jsonb->>'sys_prompt' ~ '(?<!\{)\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}(?!\})'
        )
),
updated_rows AS (
    SELECT
        "id",
        "agentBlockId",
        (
            j
            -- Update "format" if it has a single-brace placeholder
            || CASE WHEN j->>'format' ~ '(?<!\{)\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}(?!\})'
                THEN jsonb_build_object(
                    'format',
                    regexp_replace(
                        j->>'format',
                        '(?<!\{)\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}(?!\})',
                        '{{\1}}',
                        'g'
                    )
                )
                ELSE '{}'::jsonb
            END
            -- Update "prompt" if it has a single-brace placeholder
            || CASE WHEN j->>'prompt' ~ '(?<!\{)\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}(?!\})'
                THEN jsonb_build_object(
                    'prompt',
                    regexp_replace(
                        j->>'prompt',
                        '(?<!\{)\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}(?!\})',
                        '{{\1}}',
                        'g'
                    )
                )
                ELSE '{}'::jsonb
            END
            -- Update "sys_prompt" if it has a single-brace placeholder
            || CASE WHEN j->>'sys_prompt' ~ '(?<!\{)\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}(?!\})'
                THEN jsonb_build_object(
                    'sys_prompt',
                    regexp_replace(
                        j->>'sys_prompt',
                        '(?<!\{)\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}(?!\})',
                        '{{\1}}',
                        'g'
                    )
                )
                ELSE '{}'::jsonb
            END
        )::text AS "newConstantInput"
    FROM to_update
)
UPDATE "AgentNode" AS an
SET "constantInput" = ur."newConstantInput"
FROM updated_rows ur
WHERE an."id" = ur."id";

COMMIT;
