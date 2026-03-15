-- Correct credentials.provider field on all nodes with 'llm' provider credentials
UPDATE "AgentNode"
SET    "constantInput" = JSONB_SET(
         "constantInput"::jsonb,
         '{credentials,provider}',
         CASE
           WHEN "constantInput"::jsonb->'credentials'->>'id' = '53c25cb8-e3ee-465c-a4d1-e75a4c899c2a' THEN '"openai"'::jsonb
           WHEN "constantInput"::jsonb->'credentials'->>'id' = '24e5d942-d9e3-4798-8151-90143ee55629' THEN '"anthropic"'::jsonb
           WHEN "constantInput"::jsonb->'credentials'->>'id' = '4ec22295-8f97-4dd1-b42b-2c6957a02545' THEN '"groq"'::jsonb
           ELSE "constantInput"::jsonb->'credentials'->'provider'
         END
       )::text
WHERE  "constantInput"::jsonb->'credentials'->>'provider' = 'llm';
