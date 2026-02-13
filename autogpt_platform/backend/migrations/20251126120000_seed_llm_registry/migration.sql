-- Seed LLM Registry from existing hard-coded data
-- This migration populates the LlmProvider, LlmModel, and LlmModelCost tables
-- with data from the existing MODEL_METADATA and MODEL_COST dictionaries

-- Insert Providers
INSERT INTO "LlmProvider" ("id", "name", "displayName", "description", "defaultCredentialProvider", "defaultCredentialType", "supportsTools", "supportsJsonOutput", "supportsReasoning", "supportsParallelTool", "metadata")
VALUES
    (gen_random_uuid(), 'openai', 'OpenAI', 'OpenAI language models', 'openai', 'api_key', true, true, true, true, '{}'::jsonb),
    (gen_random_uuid(), 'anthropic', 'Anthropic', 'Anthropic Claude models', 'anthropic', 'api_key', true, true, true, false, '{}'::jsonb),
    (gen_random_uuid(), 'groq', 'Groq', 'Groq inference API', 'groq', 'api_key', false, true, false, false, '{}'::jsonb),
    (gen_random_uuid(), 'open_router', 'OpenRouter', 'OpenRouter unified API', 'open_router', 'api_key', true, true, false, false, '{}'::jsonb),
    (gen_random_uuid(), 'aiml_api', 'AI/ML API', 'AI/ML API models', 'aiml_api', 'api_key', false, true, false, false, '{}'::jsonb),
    (gen_random_uuid(), 'ollama', 'Ollama', 'Ollama local models', 'ollama', 'api_key', false, true, false, false, '{}'::jsonb),
    (gen_random_uuid(), 'llama_api', 'Llama API', 'Llama API models', 'llama_api', 'api_key', false, true, false, false, '{}'::jsonb),
    (gen_random_uuid(), 'v0', 'v0', 'v0 by Vercel models', 'v0', 'api_key', true, true, false, false, '{}'::jsonb)
ON CONFLICT ("name") DO NOTHING;

-- Insert Models (using CTEs to reference provider IDs)
WITH provider_ids AS (
    SELECT "id", "name" FROM "LlmProvider"
)
INSERT INTO "LlmModel" ("id", "slug", "displayName", "description", "providerId", "contextWindow", "maxOutputTokens", "isEnabled", "capabilities", "metadata")
SELECT
    gen_random_uuid(),
    model_slug,
    model_display_name,
    NULL,
    p."id",
    context_window,
    max_output_tokens,
    true,
    '{}'::jsonb,
    '{}'::jsonb
FROM (VALUES
    -- OpenAI models
    ('o3', 'O3', 'openai', 200000, 100000),
    ('o3-mini', 'O3 Mini', 'openai', 200000, 100000),
    ('o1', 'O1', 'openai', 200000, 100000),
    ('o1-mini', 'O1 Mini', 'openai', 128000, 65536),
    ('gpt-5-2025-08-07', 'GPT 5', 'openai', 400000, 128000),
    ('gpt-5.1-2025-11-13', 'GPT 5.1', 'openai', 400000, 128000),
    ('gpt-5-mini-2025-08-07', 'GPT 5 Mini', 'openai', 400000, 128000),
    ('gpt-5-nano-2025-08-07', 'GPT 5 Nano', 'openai', 400000, 128000),
    ('gpt-5-chat-latest', 'GPT 5 Chat', 'openai', 400000, 16384),
    ('gpt-4.1-2025-04-14', 'GPT 4.1', 'openai', 1000000, 32768),
    ('gpt-4.1-mini-2025-04-14', 'GPT 4.1 Mini', 'openai', 1047576, 32768),
    ('gpt-4o-mini', 'GPT 4o Mini', 'openai', 128000, 16384),
    ('gpt-4o', 'GPT 4o', 'openai', 128000, 16384),
    ('gpt-4-turbo', 'GPT 4 Turbo', 'openai', 128000, 4096),
    ('gpt-3.5-turbo', 'GPT 3.5 Turbo', 'openai', 16385, 4096),
    -- Anthropic models
    ('claude-opus-4-1-20250805', 'Claude 4.1 Opus', 'anthropic', 200000, 32000),
    ('claude-opus-4-20250514', 'Claude 4 Opus', 'anthropic', 200000, 32000),
    ('claude-sonnet-4-20250514', 'Claude 4 Sonnet', 'anthropic', 200000, 64000),
    ('claude-opus-4-5-20251101', 'Claude 4.5 Opus', 'anthropic', 200000, 64000),
    ('claude-sonnet-4-5-20250929', 'Claude 4.5 Sonnet', 'anthropic', 200000, 64000),
    ('claude-haiku-4-5-20251001', 'Claude 4.5 Haiku', 'anthropic', 200000, 64000),
    ('claude-3-7-sonnet-20250219', 'Claude 3.7 Sonnet', 'anthropic', 200000, 64000),
    ('claude-3-haiku-20240307', 'Claude 3 Haiku', 'anthropic', 200000, 4096),
    -- AI/ML API models
    ('Qwen/Qwen2.5-72B-Instruct-Turbo', 'Qwen 2.5 72B', 'aiml_api', 32000, 8000),
    ('nvidia/llama-3.1-nemotron-70b-instruct', 'Llama 3.1 Nemotron 70B', 'aiml_api', 128000, 40000),
    ('meta-llama/Llama-3.3-70B-Instruct-Turbo', 'Llama 3.3 70B', 'aiml_api', 128000, NULL),
    ('meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo', 'Meta Llama 3.1 70B', 'aiml_api', 131000, 2000),
    ('meta-llama/Llama-3.2-3B-Instruct-Turbo', 'Llama 3.2 3B', 'aiml_api', 128000, NULL),
    -- Groq models
    ('llama-3.3-70b-versatile', 'Llama 3.3 70B', 'groq', 128000, 32768),
    ('llama-3.1-8b-instant', 'Llama 3.1 8B', 'groq', 128000, 8192),
    -- Ollama models
    ('llama3.3', 'Llama 3.3', 'ollama', 8192, NULL),
    ('llama3.2', 'Llama 3.2', 'ollama', 8192, NULL),
    ('llama3', 'Llama 3', 'ollama', 8192, NULL),
    ('llama3.1:405b', 'Llama 3.1 405B', 'ollama', 8192, NULL),
    ('dolphin-mistral:latest', 'Dolphin Mistral', 'ollama', 32768, NULL),
    -- OpenRouter models
    ('google/gemini-2.5-pro-preview-03-25', 'Gemini 2.5 Pro', 'open_router', 1050000, 8192),
    ('google/gemini-3-pro-preview', 'Gemini 3 Pro Preview', 'open_router', 1048576, 65535),
    ('google/gemini-2.5-flash', 'Gemini 2.5 Flash', 'open_router', 1048576, 65535),
    ('google/gemini-2.0-flash-001', 'Gemini 2.0 Flash', 'open_router', 1048576, 8192),
    ('google/gemini-2.5-flash-lite-preview-06-17', 'Gemini 2.5 Flash Lite Preview', 'open_router', 1048576, 65535),
    ('google/gemini-2.0-flash-lite-001', 'Gemini 2.0 Flash Lite', 'open_router', 1048576, 8192),
    ('mistralai/mistral-nemo', 'Mistral Nemo', 'open_router', 128000, 4096),
    ('cohere/command-r-08-2024', 'Command R', 'open_router', 128000, 4096),
    ('cohere/command-r-plus-08-2024', 'Command R Plus', 'open_router', 128000, 4096),
    ('deepseek/deepseek-chat', 'DeepSeek Chat', 'open_router', 64000, 2048),
    ('deepseek/deepseek-r1-0528', 'DeepSeek R1', 'open_router', 163840, 163840),
    ('perplexity/sonar', 'Perplexity Sonar', 'open_router', 127000, 8000),
    ('perplexity/sonar-pro', 'Perplexity Sonar Pro', 'open_router', 200000, 8000),
    ('perplexity/sonar-deep-research', 'Perplexity Sonar Deep Research', 'open_router', 128000, 16000),
    ('nousresearch/hermes-3-llama-3.1-405b', 'Hermes 3 Llama 3.1 405B', 'open_router', 131000, 4096),
    ('nousresearch/hermes-3-llama-3.1-70b', 'Hermes 3 Llama 3.1 70B', 'open_router', 12288, 12288),
    ('openai/gpt-oss-120b', 'GPT OSS 120B', 'open_router', 131072, 131072),
    ('openai/gpt-oss-20b', 'GPT OSS 20B', 'open_router', 131072, 32768),
    ('amazon/nova-lite-v1', 'Amazon Nova Lite', 'open_router', 300000, 5120),
    ('amazon/nova-micro-v1', 'Amazon Nova Micro', 'open_router', 128000, 5120),
    ('amazon/nova-pro-v1', 'Amazon Nova Pro', 'open_router', 300000, 5120),
    ('microsoft/wizardlm-2-8x22b', 'WizardLM 2 8x22B', 'open_router', 65536, 4096),
    ('gryphe/mythomax-l2-13b', 'MythoMax L2 13B', 'open_router', 4096, 4096),
    ('meta-llama/llama-4-scout', 'Llama 4 Scout', 'open_router', 131072, 131072),
    ('meta-llama/llama-4-maverick', 'Llama 4 Maverick', 'open_router', 1048576, 1000000),
    ('x-ai/grok-4', 'Grok 4', 'open_router', 256000, 256000),
    ('x-ai/grok-4-fast', 'Grok 4 Fast', 'open_router', 2000000, 30000),
    ('x-ai/grok-4.1-fast', 'Grok 4.1 Fast', 'open_router', 2000000, 30000),
    ('x-ai/grok-code-fast-1', 'Grok Code Fast 1', 'open_router', 256000, 10000),
    ('moonshotai/kimi-k2', 'Kimi K2', 'open_router', 131000, 131000),
    ('qwen/qwen3-235b-a22b-thinking-2507', 'Qwen 3 235B Thinking', 'open_router', 262144, 262144),
    ('qwen/qwen3-coder', 'Qwen 3 Coder', 'open_router', 262144, 262144),
    -- Llama API models
    ('Llama-4-Scout-17B-16E-Instruct-FP8', 'Llama 4 Scout', 'llama_api', 128000, 4028),
    ('Llama-4-Maverick-17B-128E-Instruct-FP8', 'Llama 4 Maverick', 'llama_api', 128000, 4028),
    ('Llama-3.3-8B-Instruct', 'Llama 3.3 8B', 'llama_api', 128000, 4028),
    ('Llama-3.3-70B-Instruct', 'Llama 3.3 70B', 'llama_api', 128000, 4028),
    -- v0 models
    ('v0-1.5-md', 'v0 1.5 MD', 'v0', 128000, 64000),
    ('v0-1.5-lg', 'v0 1.5 LG', 'v0', 512000, 64000),
    ('v0-1.0-md', 'v0 1.0 MD', 'v0', 128000, 64000)
) AS models(model_slug, model_display_name, provider_name, context_window, max_output_tokens)
JOIN provider_ids p ON p."name" = models.provider_name
ON CONFLICT ("slug") DO NOTHING;

-- Insert Costs (using CTEs to reference model IDs)
WITH model_ids AS (
    SELECT "id", "slug", "providerId" FROM "LlmModel"
),
provider_ids AS (
    SELECT "id", "name" FROM "LlmProvider"
)
INSERT INTO "LlmModelCost" ("id", "unit", "creditCost", "credentialProvider", "credentialId", "credentialType", "currency", "metadata", "llmModelId")
SELECT
    gen_random_uuid(),
    'RUN'::"LlmCostUnit",
    cost,
    p."name",
    NULL,
    'api_key',
    NULL,
    '{}'::jsonb,
    m."id"
FROM (VALUES
    -- OpenAI costs
    ('o3', 4),
    ('o3-mini', 2),
    ('o1', 16),
    ('o1-mini', 4),
    ('gpt-5-2025-08-07', 2),
    ('gpt-5.1-2025-11-13', 5),
    ('gpt-5-mini-2025-08-07', 1),
    ('gpt-5-nano-2025-08-07', 1),
    ('gpt-5-chat-latest', 5),
    ('gpt-4.1-2025-04-14', 2),
    ('gpt-4.1-mini-2025-04-14', 1),
    ('gpt-4o-mini', 1),
    ('gpt-4o', 3),
    ('gpt-4-turbo', 10),
    ('gpt-3.5-turbo', 1),
    -- Anthropic costs
    ('claude-opus-4-1-20250805', 21),
    ('claude-opus-4-20250514', 21),
    ('claude-sonnet-4-20250514', 5),
    ('claude-haiku-4-5-20251001', 4),
    ('claude-opus-4-5-20251101', 14),
    ('claude-sonnet-4-5-20250929', 9),
    ('claude-3-7-sonnet-20250219', 5),
    ('claude-3-haiku-20240307', 1),
    -- AI/ML API costs
    ('Qwen/Qwen2.5-72B-Instruct-Turbo', 1),
    ('nvidia/llama-3.1-nemotron-70b-instruct', 1),
    ('meta-llama/Llama-3.3-70B-Instruct-Turbo', 1),
    ('meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo', 1),
    ('meta-llama/Llama-3.2-3B-Instruct-Turbo', 1),
    -- Groq costs
    ('llama-3.3-70b-versatile', 1),
    ('llama-3.1-8b-instant', 1),
    -- Ollama costs
    ('llama3.3', 1),
    ('llama3.2', 1),
    ('llama3', 1),
    ('llama3.1:405b', 1),
    ('dolphin-mistral:latest', 1),
    -- OpenRouter costs
    ('google/gemini-2.5-pro-preview-03-25', 4),
    ('google/gemini-3-pro-preview', 5),
    ('mistralai/mistral-nemo', 1),
    ('cohere/command-r-08-2024', 1),
    ('cohere/command-r-plus-08-2024', 3),
    ('deepseek/deepseek-chat', 2),
    ('perplexity/sonar', 1),
    ('perplexity/sonar-pro', 5),
    ('perplexity/sonar-deep-research', 10),
    ('nousresearch/hermes-3-llama-3.1-405b', 1),
    ('nousresearch/hermes-3-llama-3.1-70b', 1),
    ('amazon/nova-lite-v1', 1),
    ('amazon/nova-micro-v1', 1),
    ('amazon/nova-pro-v1', 1),
    ('microsoft/wizardlm-2-8x22b', 1),
    ('gryphe/mythomax-l2-13b', 1),
    ('meta-llama/llama-4-scout', 1),
    ('meta-llama/llama-4-maverick', 1),
    ('x-ai/grok-4', 9),
    ('x-ai/grok-4-fast', 1),
    ('x-ai/grok-4.1-fast', 1),
    ('x-ai/grok-code-fast-1', 1),
    ('moonshotai/kimi-k2', 1),
    ('qwen/qwen3-235b-a22b-thinking-2507', 1),
    ('qwen/qwen3-coder', 9),
    ('google/gemini-2.5-flash', 1),
    ('google/gemini-2.0-flash-001', 1),
    ('google/gemini-2.5-flash-lite-preview-06-17', 1),
    ('google/gemini-2.0-flash-lite-001', 1),
    ('deepseek/deepseek-r1-0528', 1),
    ('openai/gpt-oss-120b', 1),
    ('openai/gpt-oss-20b', 1),
    -- Llama API costs
    ('Llama-4-Scout-17B-16E-Instruct-FP8', 1),
    ('Llama-4-Maverick-17B-128E-Instruct-FP8', 1),
    ('Llama-3.3-8B-Instruct', 1),
    ('Llama-3.3-70B-Instruct', 1),
    -- v0 costs
    ('v0-1.5-md', 1),
    ('v0-1.5-lg', 2),
    ('v0-1.0-md', 1)
) AS costs(model_slug, cost)
JOIN model_ids m ON m."slug" = costs.model_slug
JOIN provider_ids p ON p."id" = m."providerId"
ON CONFLICT ("llmModelId", "credentialProvider", "unit") DO NOTHING;

