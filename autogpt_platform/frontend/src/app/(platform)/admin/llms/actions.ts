"use server";

import BackendApi from "@/lib/autogpt-server-api";
import type {
  CreateLlmModelRequest,
  LlmCreatorsResponse,
  LlmMigrationsResponse,
  LlmModelsResponse,
  LlmProvidersResponse,
  ToggleLlmModelRequest,
  UpdateLlmModelRequest,
  UpsertLlmCreatorRequest,
  UpsertLlmProviderRequest,
} from "@/lib/autogpt-server-api/types";
import { revalidatePath } from "next/cache";

const ADMIN_LLM_PATH = "/admin/llms";

export async function fetchLlmProviders(): Promise<LlmProvidersResponse> {
  const api = new BackendApi();
  return await api.listAdminLlmProviders(true);
}

export async function fetchLlmModels(): Promise<LlmModelsResponse> {
  const api = new BackendApi();
  return await api.listAdminLlmModels();
}

export async function createLlmProviderAction(formData: FormData) {
  const payload: UpsertLlmProviderRequest = {
    name: String(formData.get("name") || "").trim(),
    display_name: String(formData.get("display_name") || "").trim(),
    description: formData.get("description")
      ? String(formData.get("description"))
      : undefined,
    default_credential_provider: formData.get("default_credential_provider")
      ? String(formData.get("default_credential_provider")).trim()
      : undefined,
    default_credential_id: undefined, // Not needed - system uses credential_provider to lookup
    default_credential_type: "api_key", // Default to api_key
    supports_tools: formData.get("supports_tools") === "on",
    supports_json_output: formData.get("supports_json_output") !== "off",
    supports_reasoning: formData.get("supports_reasoning") === "on",
    supports_parallel_tool: formData.get("supports_parallel_tool") === "on",
    metadata: {},
  };

  const api = new BackendApi();
  await api.createAdminLlmProvider(payload);
  revalidatePath(ADMIN_LLM_PATH);
}

export async function createLlmModelAction(formData: FormData) {
  const providerId = String(formData.get("provider_id"));
  const creatorId = formData.get("creator_id");

  // Fetch provider to get default credentials
  const api = new BackendApi();
  const providersResponse = await api.listAdminLlmProviders(false);
  const provider = providersResponse.providers.find((p) => p.id === providerId);

  if (!provider) {
    throw new Error("Provider not found");
  }

  const payload: CreateLlmModelRequest = {
    slug: String(formData.get("slug") || "").trim(),
    display_name: String(formData.get("display_name") || "").trim(),
    description: formData.get("description")
      ? String(formData.get("description"))
      : undefined,
    provider_id: providerId,
    creator_id: creatorId ? String(creatorId) : undefined,
    context_window: Number(formData.get("context_window") || 0),
    max_output_tokens: formData.get("max_output_tokens")
      ? Number(formData.get("max_output_tokens"))
      : undefined,
    is_enabled: formData.get("is_enabled") !== "off",
    capabilities: {},
    metadata: {},
    costs: [
      {
        credit_cost: Number(formData.get("credit_cost") || 0),
        credential_provider:
          provider.default_credential_provider || provider.name,
        credential_id: provider.default_credential_id || undefined,
        credential_type: provider.default_credential_type || "api_key",
        metadata: {},
      },
    ],
  };

  await api.createAdminLlmModel(payload);
  revalidatePath(ADMIN_LLM_PATH);
}

export async function updateLlmModelAction(formData: FormData) {
  const modelId = String(formData.get("model_id"));
  const creatorId = formData.get("creator_id");

  const payload: UpdateLlmModelRequest = {
    display_name: formData.get("display_name")
      ? String(formData.get("display_name"))
      : undefined,
    description: formData.get("description")
      ? String(formData.get("description"))
      : undefined,
    provider_id: formData.get("provider_id")
      ? String(formData.get("provider_id"))
      : undefined,
    creator_id: creatorId ? String(creatorId) : undefined,
    context_window: formData.get("context_window")
      ? Number(formData.get("context_window"))
      : undefined,
    max_output_tokens: formData.get("max_output_tokens")
      ? Number(formData.get("max_output_tokens"))
      : undefined,
    is_enabled: formData.get("is_enabled")
      ? formData.get("is_enabled") === "on"
      : undefined,
    costs: formData.get("credit_cost")
      ? [
          {
            credit_cost: Number(formData.get("credit_cost")),
            credential_provider: String(
              formData.get("credential_provider") || "",
            ).trim(),
            credential_id: formData.get("credential_id")
              ? String(formData.get("credential_id"))
              : undefined,
            credential_type: formData.get("credential_type")
              ? String(formData.get("credential_type"))
              : undefined,
            metadata: {},
          },
        ]
      : undefined,
  };

  const api = new BackendApi();
  await api.updateAdminLlmModel(modelId, payload);
  revalidatePath(ADMIN_LLM_PATH);
}

export async function toggleLlmModelAction(formData: FormData): Promise<void> {
  const modelId = String(formData.get("model_id"));
  const shouldEnable = formData.get("is_enabled") === "true";
  const migrateToSlug = formData.get("migrate_to_slug");
  const migrationReason = formData.get("migration_reason");
  const customCreditCost = formData.get("custom_credit_cost");

  const payload: ToggleLlmModelRequest = {
    is_enabled: shouldEnable,
    migrate_to_slug: migrateToSlug ? String(migrateToSlug) : undefined,
    migration_reason: migrationReason ? String(migrationReason) : undefined,
    custom_credit_cost: customCreditCost ? Number(customCreditCost) : undefined,
  };
  const api = new BackendApi();
  await api.toggleAdminLlmModel(modelId, payload);
  revalidatePath(ADMIN_LLM_PATH);
}

export async function deleteLlmModelAction(formData: FormData) {
  try {
    const modelId = String(formData.get("model_id"));
    const replacementModelSlug = String(formData.get("replacement_model_slug"));

    if (!replacementModelSlug) {
      throw new Error("Replacement model is required");
    }

    const api = new BackendApi();
    const result = await api.deleteAdminLlmModel(modelId, replacementModelSlug);
    revalidatePath(ADMIN_LLM_PATH);
    return result;
  } catch (error) {
    console.error("Delete model error:", error);
    throw error instanceof Error ? error : new Error("Failed to delete model");
  }
}

// Migration management actions
export async function fetchLlmMigrations(
  includeReverted: boolean = false
): Promise<LlmMigrationsResponse> {
  const api = new BackendApi();
  return await api.listAdminLlmMigrations(includeReverted);
}

export async function revertLlmMigrationAction(
  formData: FormData
): Promise<void> {
  try {
    const migrationId = String(formData.get("migration_id"));
    const api = new BackendApi();
    await api.revertAdminLlmMigration(migrationId);
    revalidatePath(ADMIN_LLM_PATH);
  } catch (error) {
    console.error("Revert migration error:", error);
    throw error instanceof Error
      ? error
      : new Error("Failed to revert migration");
  }
}

// Creator management actions
export async function fetchLlmCreators(): Promise<LlmCreatorsResponse> {
  const api = new BackendApi();
  return await api.listAdminLlmCreators();
}

export async function createLlmCreatorAction(formData: FormData): Promise<void> {
  const payload: UpsertLlmCreatorRequest = {
    name: String(formData.get("name") || "").trim(),
    display_name: String(formData.get("display_name") || "").trim(),
    description: formData.get("description")
      ? String(formData.get("description"))
      : undefined,
    website_url: formData.get("website_url")
      ? String(formData.get("website_url")).trim()
      : undefined,
    logo_url: formData.get("logo_url")
      ? String(formData.get("logo_url")).trim()
      : undefined,
    metadata: {},
  };

  const api = new BackendApi();
  await api.createAdminLlmCreator(payload);
  revalidatePath(ADMIN_LLM_PATH);
}

export async function updateLlmCreatorAction(formData: FormData): Promise<void> {
  const creatorId = String(formData.get("creator_id"));
  const payload: UpsertLlmCreatorRequest = {
    name: String(formData.get("name") || "").trim(),
    display_name: String(formData.get("display_name") || "").trim(),
    description: formData.get("description")
      ? String(formData.get("description"))
      : undefined,
    website_url: formData.get("website_url")
      ? String(formData.get("website_url")).trim()
      : undefined,
    logo_url: formData.get("logo_url")
      ? String(formData.get("logo_url")).trim()
      : undefined,
    metadata: {},
  };

  const api = new BackendApi();
  await api.updateAdminLlmCreator(creatorId, payload);
  revalidatePath(ADMIN_LLM_PATH);
}

export async function deleteLlmCreatorAction(formData: FormData): Promise<void> {
  try {
    const creatorId = String(formData.get("creator_id"));
    const api = new BackendApi();
    await api.deleteAdminLlmCreator(creatorId);
    revalidatePath(ADMIN_LLM_PATH);
  } catch (error) {
    console.error("Delete creator error:", error);
    throw error instanceof Error ? error : new Error("Failed to delete creator");
  }
}

