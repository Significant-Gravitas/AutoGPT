"use server";

import { revalidatePath } from "next/cache";
import {
  createRequestHeaders,
  getServerAuthToken,
} from "@/lib/autogpt-server-api/helpers";
import { environment } from "@/services/environment";

const ADMIN_LLM_PATH = "/admin/llms";

// =============================================================================
// Authenticated Fetch Helper
// =============================================================================

async function adminFetch(
  endpoint: string,
  options: RequestInit = {},
): Promise<{ status: number; data: any }> {
  const baseUrl = environment.getAGPTServerBaseUrl();
  const token = await getServerAuthToken();
  const headers = createRequestHeaders(
    token,
    !!options.body,
    "application/json",
  );

  const response = await fetch(`${baseUrl}${endpoint}`, {
    ...options,
    headers: {
      ...headers,
      ...((options.headers as Record<string, string>) || {}),
    },
  });

  let data: any = null;
  if (response.status !== 204) {
    const contentType = response.headers.get("content-type");
    const text = await response.text();
    if (text && contentType?.includes("application/json")) {
      try {
        data = JSON.parse(text);
      } catch {
        data = text;
      }
    } else {
      data = text;
    }
  }

  if (!response.ok) {
    const errorMessage =
      data?.detail || data?.message || `HTTP ${response.status}`;
    throw new Error(errorMessage);
  }

  return { status: response.status, data };
}

// =============================================================================
// Utilities
// =============================================================================

function getRequiredFormField(
  formData: FormData,
  fieldName: string,
  displayName?: string,
): string {
  const raw = formData.get(fieldName);
  const value = raw ? String(raw).trim() : "";
  if (!value) {
    throw new Error(`${displayName || fieldName} is required`);
  }
  return value;
}

function getRequiredPositiveNumber(
  formData: FormData,
  fieldName: string,
  displayName?: string,
): number {
  const raw = formData.get(fieldName);
  const value = Number(raw);
  if (raw === null || raw === "" || !Number.isFinite(value) || value <= 0) {
    throw new Error(`${displayName || fieldName} must be a positive number`);
  }
  return value;
}

function getRequiredNumber(
  formData: FormData,
  fieldName: string,
  displayName?: string,
): number {
  const raw = formData.get(fieldName);
  const value = Number(raw);
  if (raw === null || raw === "" || !Number.isFinite(value)) {
    throw new Error(`${displayName || fieldName} is required`);
  }
  return value;
}

// =============================================================================
// Provider Actions
// =============================================================================

export async function fetchLlmProviders() {
  const { data } = await adminFetch("/api/llm/admin/providers");
  return data;
}

export async function createLlmProviderAction(formData: FormData) {
  const payload = {
    name: String(formData.get("name") || "").trim(),
    display_name: String(formData.get("display_name") || "").trim(),
    description: formData.get("description")
      ? String(formData.get("description"))
      : undefined,
    default_credential_provider: formData.get("default_credential_provider")
      ? String(formData.get("default_credential_provider")).trim()
      : undefined,
    default_credential_id: formData.get("default_credential_id")
      ? String(formData.get("default_credential_id")).trim()
      : undefined,
    default_credential_type: formData.get("default_credential_type")
      ? String(formData.get("default_credential_type")).trim()
      : "api_key",
    metadata: {},
  };

  await adminFetch("/api/llm/providers", {
    method: "POST",
    body: JSON.stringify(payload),
  });
  revalidatePath(ADMIN_LLM_PATH);
}

export async function deleteLlmProviderAction(
  formData: FormData,
): Promise<void> {
  const providerName = getRequiredFormField(
    formData,
    "provider_id",
    "Provider",
  );
  await adminFetch(`/api/llm/providers/${providerName}`, { method: "DELETE" });
  revalidatePath(ADMIN_LLM_PATH);
}

export async function updateLlmProviderAction(formData: FormData) {
  const providerName = getRequiredFormField(
    formData,
    "provider_id",
    "Provider",
  );

  const payload = {
    display_name: String(formData.get("display_name") || "").trim(),
    description: formData.get("description")
      ? String(formData.get("description"))
      : undefined,
    default_credential_provider: formData.get("default_credential_provider")
      ? String(formData.get("default_credential_provider")).trim()
      : undefined,
    default_credential_id: formData.get("default_credential_id")
      ? String(formData.get("default_credential_id")).trim()
      : undefined,
    default_credential_type: formData.get("default_credential_type")
      ? String(formData.get("default_credential_type")).trim()
      : "api_key",
    metadata: {},
  };

  await adminFetch(`/api/llm/providers/${providerName}`, {
    method: "PATCH",
    body: JSON.stringify(payload),
  });
  revalidatePath(ADMIN_LLM_PATH);
}

// =============================================================================
// Model Actions
// =============================================================================

export async function fetchLlmModels(page?: number, pageSize?: number) {
  const params = new URLSearchParams();
  if (page) params.set("page", String(page));
  if (pageSize) params.set("page_size", String(pageSize));
  params.set("enabled_only", "false");
  const query = params.toString() ? `?${params.toString()}` : "";
  const { data } = await adminFetch(`/api/llm/admin/models${query}`);
  return data;
}

export async function createLlmModelAction(formData: FormData) {
  const creditCost = getRequiredNumber(formData, "credit_cost", "Credit cost");

  const payload = {
    slug: String(formData.get("slug") || "").trim(),
    display_name: String(formData.get("display_name") || "").trim(),
    description: formData.get("description")
      ? String(formData.get("description"))
      : undefined,
    provider_id: getRequiredFormField(formData, "provider_id", "Provider"),
    creator_id: formData.get("creator_id")
      ? String(formData.get("creator_id"))
      : undefined,
    context_window: getRequiredPositiveNumber(
      formData,
      "context_window",
      "Context window",
    ),
    max_output_tokens: formData.get("max_output_tokens")
      ? Number(formData.get("max_output_tokens"))
      : undefined,
    price_tier: Number(formData.get("price_tier") || 1),
    is_enabled: formData.getAll("is_enabled").includes("on"),
    capabilities: {},
    metadata: {},
    costs: [
      {
        unit: String(formData.get("unit") || "RUN"),
        credit_cost: creditCost,
        metadata: {},
      },
    ],
  };

  await adminFetch("/api/llm/models", {
    method: "POST",
    body: JSON.stringify(payload),
  });
  revalidatePath(ADMIN_LLM_PATH);
}

export async function updateLlmModelAction(formData: FormData) {
  const modelSlug = getRequiredFormField(formData, "model_id", "Model");

  const payload: Record<string, any> = {};

  if (formData.get("display_name"))
    payload.display_name = String(formData.get("display_name"));
  if (formData.get("description"))
    payload.description = String(formData.get("description"));
  if (formData.get("provider_id"))
    payload.provider_id = String(formData.get("provider_id"));
  if (formData.get("creator_id"))
    payload.creator_id = String(formData.get("creator_id"));
  if (formData.get("context_window"))
    payload.context_window = Number(formData.get("context_window"));
  if (formData.get("max_output_tokens"))
    payload.max_output_tokens = Number(formData.get("max_output_tokens"));
  if (formData.has("is_enabled"))
    payload.is_enabled = formData.getAll("is_enabled").includes("on");

  await adminFetch(`/api/llm/models/${modelSlug}`, {
    method: "PATCH",
    body: JSON.stringify(payload),
  });
  revalidatePath(ADMIN_LLM_PATH);
}

export async function toggleLlmModelAction(formData: FormData): Promise<void> {
  const modelSlug = getRequiredFormField(formData, "model_id", "Model");
  const shouldEnable = formData.get("is_enabled") === "true";

  const payload: Record<string, any> = { is_enabled: shouldEnable };

  // Migration params (only when disabling)
  if (!shouldEnable) {
    const migrateToSlug = formData.get("migrate_to_slug");
    if (migrateToSlug) payload.migrate_to_slug = String(migrateToSlug);
    const reason = formData.get("migration_reason");
    if (reason) payload.migration_reason = String(reason);
    const customCost = formData.get("custom_credit_cost");
    if (customCost) payload.custom_credit_cost = Number(customCost);
  }

  await adminFetch(`/api/llm/models/${modelSlug}/toggle`, {
    method: "POST",
    body: JSON.stringify(payload),
  });
  revalidatePath(ADMIN_LLM_PATH);
}

export async function deleteLlmModelAction(formData: FormData): Promise<void> {
  const modelSlug = getRequiredFormField(formData, "model_id", "Model");
  const replacementSlug = formData.get("replacement_model_slug");
  const params = new URLSearchParams();
  if (replacementSlug)
    params.set("replacement_model_slug", String(replacementSlug));
  const query = params.toString() ? `?${params.toString()}` : "";
  await adminFetch(`/api/llm/models/${modelSlug}${query}`, {
    method: "DELETE",
  });
  revalidatePath(ADMIN_LLM_PATH);
}

export async function fetchLlmModelUsage(modelSlug: string) {
  const { data } = await adminFetch(`/api/llm/models/${modelSlug}/usage`);
  return data;
}

// =============================================================================
// Migration Actions
// =============================================================================

export async function fetchLlmMigrations(includeReverted: boolean = false) {
  const params = new URLSearchParams();
  if (includeReverted) params.set("include_reverted", "true");
  const query = params.toString() ? `?${params.toString()}` : "";
  const { data } = await adminFetch(`/api/llm/migrations${query}`);
  return data;
}

export async function revertLlmMigrationAction(
  formData: FormData,
): Promise<void> {
  const migrationId = getRequiredFormField(
    formData,
    "migration_id",
    "Migration",
  );
  await adminFetch(`/api/llm/migrations/${migrationId}/revert`, {
    method: "POST",
  });
  revalidatePath(ADMIN_LLM_PATH);
}

// =============================================================================
// Creator Actions
// =============================================================================

export async function fetchLlmCreators() {
  const { data } = await adminFetch(`/api/llm/creators`);
  return data;
}

export async function createLlmCreatorAction(
  formData: FormData,
): Promise<void> {
  const payload = {
    name: String(formData.get("name") || "").trim(),
    display_name: String(formData.get("display_name") || "").trim(),
    description: formData.get("description")
      ? String(formData.get("description"))
      : undefined,
    website_url: formData.get("website_url")
      ? String(formData.get("website_url"))
      : undefined,
    metadata: {},
  };

  await adminFetch("/api/llm/creators", {
    method: "POST",
    body: JSON.stringify(payload),
  });
  revalidatePath(ADMIN_LLM_PATH);
}

export async function updateLlmCreatorAction(
  formData: FormData,
): Promise<void> {
  const creatorName = getRequiredFormField(formData, "creator_id", "Creator");

  const payload: Record<string, any> = {};
  if (formData.get("display_name"))
    payload.display_name = String(formData.get("display_name"));
  if (formData.get("description"))
    payload.description = String(formData.get("description"));
  if (formData.get("website_url"))
    payload.website_url = String(formData.get("website_url"));

  await adminFetch(`/api/llm/creators/${creatorName}`, {
    method: "PATCH",
    body: JSON.stringify(payload),
  });
  revalidatePath(ADMIN_LLM_PATH);
}

export async function deleteLlmCreatorAction(
  formData: FormData,
): Promise<void> {
  const creatorName = getRequiredFormField(formData, "creator_id", "Creator");
  await adminFetch(`/api/llm/creators/${creatorName}`, { method: "DELETE" });
  revalidatePath(ADMIN_LLM_PATH);
}

// =============================================================================
// Recommended Model Actions
// =============================================================================

export async function setRecommendedModelAction(
  formData: FormData,
): Promise<void> {
  const modelSlug = getRequiredFormField(formData, "model_id", "Model");

  // Set recommended by updating the model
  await adminFetch(`/api/llm/models/${modelSlug}`, {
    method: "PATCH",
    body: JSON.stringify({ is_recommended: true }),
  });
  revalidatePath(ADMIN_LLM_PATH);
}
