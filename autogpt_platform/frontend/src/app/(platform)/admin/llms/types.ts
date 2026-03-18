// Type definitions for LLM registry admin UI
// These match the API response formats from our admin endpoints

export interface LlmProvider {
  id: string;
  name: string;
  display_name: string;
  description: string | null;
  default_credential_provider: string | null;
  default_credential_id: string | null;
  default_credential_type: string | null;
  metadata: Record<string, any>;
  created_at: string | null;
  updated_at: string | null;
  models?: LlmModel[];
}

export interface LlmModel {
  id: string;
  slug: string;
  display_name: string;
  description: string | null;
  provider_id: string;
  creator_id: string | null;
  context_window: number;
  max_output_tokens: number | null;
  price_tier: number;
  is_enabled: boolean;
  is_recommended: boolean;
  supports_tools: boolean;
  supports_json_output: boolean;
  supports_reasoning: boolean;
  supports_parallel_tool_calls: boolean;
  capabilities: Record<string, any>;
  metadata: Record<string, any>;
  created_at: string | null;
  updated_at: string | null;
}

export interface LlmModelCreator {
  id: string;
  name: string;
  display_name: string;
  description: string | null;
  website_url: string | null;
  logo_url: string | null;
  metadata: Record<string, any>;
}

export interface LlmModelMigration {
  id: string;
  source_model_slug: string;
  target_model_slug: string;
  reason: string | null;
  migrated_node_ids: any[];
  node_count: number;
  custom_credit_cost: number | null;
  is_reverted: boolean;
  reverted_at: string | null;
  created_at: string;
  updated_at: string;
}
