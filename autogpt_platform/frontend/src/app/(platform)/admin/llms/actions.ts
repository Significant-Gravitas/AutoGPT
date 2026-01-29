"use server";

import { revalidatePath } from "next/cache";

// Generated API functions
import {
  getV2ListLlmProviders,
  postV2CreateLlmProvider,
  patchV2UpdateLlmProvider,
  deleteV2DeleteLlmProvider,
  getV2ListLlmModels,
  postV2CreateLlmModel,
  patchV2UpdateLlmModel,
  patchV2ToggleLlmModelAvailability,
  deleteV2DeleteLlmModelAndMigrateWorkflows,
  getV2GetModelUsageCount,
  getV2ListModelMigrations,
  postV2RevertAModelMigration,
  getV2ListModelCreators,
  postV2CreateModelCreator,
  patchV2UpdateModelCreator,
  deleteV2DeleteModelCreator,
  postV2SetRecommendedModel,
} from "@/app/api/__generated__/endpoints/admin/admin";

// Generated types
import type { LlmProvidersResponse } from "@/app/api/__generated__/models/llmProvidersResponse";
import type { LlmModelsResponse } from "@/app/api/__generated__/models/llmModelsResponse";
import type { UpsertLlmProviderRequest } from "@/app/api/__generated__/models/upsertLlmProviderRequest";
import type { CreateLlmModelRequest } from "@/app/api/__generated__/models/createLlmModelRequest";
import type { UpdateLlmModelRequest } from "@/app/api/__generated__/models/updateLlmModelRequest";
import type { ToggleLlmModelRequest } from "@/app/api/__generated__/models/toggleLlmModelRequest";
import type { LlmMigrationsResponse } from "@/app/api/__generated__/models/llmMigrationsResponse";
import type { LlmCreatorsResponse } from "@/app/api/__generated__/models/llmCreatorsResponse";
import type { UpsertLlmCreatorRequest } from "@/app/api/__generated__/models/upsertLlmCreatorRequest";
import type { LlmModelUsageResponse } from "@/app/api/__generated__/models/llmModelUsageResponse";
import { LlmCostUnit } from "@/app/api/__generated__/models/llmCostUnit";

const ADMIN_LLM_PATH = "/admin/llms";

// =============================================================================
// Utilities
// =============================================================================

/**
 * Extracts and validates a required string field from FormData.
 * Throws an error if the field is missing or empty.
 */
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

/**
 * Extracts and validates a required positive number field from FormData.
 * Throws an error if the field is missing, empty, or not a positive number.
 */
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

/**
 * Extracts and validates a required number field from FormData.
 * Throws an error if the field is missing, empty, or not a finite number.
 */
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

export async function fetchLlmProviders(): Promise<LlmProvidersResponse> {
  const response = await getV2ListLlmProviders({ include_models: true });
  if (response.status !== 200) {
    throw new Error("Failed to fetch LLM providers");
  }
  return response.data;
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
    default_credential_id: formData.get("default_credential_id")
      ? String(formData.get("default_credential_id")).trim()
      : undefined,
    default_credential_type: formData.get("default_credential_type")
      ? String(formData.get("default_credential_type")).trim()
      : "api_key",
    supports_tools: formData.getAll("supports_tools").includes("on"),
    supports_json_output: formData
      .getAll("supports_json_output")
      .includes("on"),
    supports_reasoning: formData.getAll("supports_reasoning").includes("on"),
    supports_parallel_tool: formData
      .getAll("supports_parallel_tool")
      .includes("on"),
    metadata: {},
  };

  const response = await postV2CreateLlmProvider(payload);
  if (response.status !== 200) {
    throw new Error("Failed to create LLM provider");
  }
  revalidatePath(ADMIN_LLM_PATH);
}

export async function deleteLlmProviderAction(
  formData: FormData,
): Promise<void> {
  const providerId = getRequiredFormField(
    formData,
    "provider_id",
    "Provider id",
  );

  const response = await deleteV2DeleteLlmProvider(providerId);
  if (response.status !== 200) {
    const errorData = response.data as { detail?: string };
    throw new Error(errorData?.detail || "Failed to delete provider");
  }
  revalidatePath(ADMIN_LLM_PATH);
}

export async function updateLlmProviderAction(formData: FormData) {
  const providerId = getRequiredFormField(
    formData,
    "provider_id",
    "Provider id",
  );

  const payload: UpsertLlmProviderRequest = {
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
    supports_tools: formData.getAll("supports_tools").includes("on"),
    supports_json_output: formData
      .getAll("supports_json_output")
      .includes("on"),
    supports_reasoning: formData.getAll("supports_reasoning").includes("on"),
    supports_parallel_tool: formData
      .getAll("supports_parallel_tool")
      .includes("on"),
    metadata: {},
  };

  const response = await patchV2UpdateLlmProvider(providerId, payload);
  if (response.status !== 200) {
    throw new Error("Failed to update LLM provider");
  }
  revalidatePath(ADMIN_LLM_PATH);
}

// =============================================================================
// Model Actions
// =============================================================================

export async function fetchLlmModels(): Promise<LlmModelsResponse> {
  const response = await getV2ListLlmModels();
  if (response.status !== 200) {
    throw new Error("Failed to fetch LLM models");
  }
  return response.data;
}

export async function createLlmModelAction(formData: FormData) {
  const providerId = getRequiredFormField(formData, "provider_id", "Provider");
  const creatorId = formData.get("creator_id");
  const contextWindow = getRequiredPositiveNumber(
    formData,
    "context_window",
    "Context window",
  );
  const creditCost = getRequiredNumber(formData, "credit_cost", "Credit cost");

  // Fetch provider to get default credentials
  const providersResponse = await getV2ListLlmProviders({
    include_models: false,
  });
  if (providersResponse.status !== 200) {
    throw new Error("Failed to fetch providers");
  }
  const provider = providersResponse.data.providers.find(
    (p) => p.id === providerId,
  );

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
    context_window: contextWindow,
    max_output_tokens: formData.get("max_output_tokens")
      ? Number(formData.get("max_output_tokens"))
      : undefined,
    is_enabled: formData.getAll("is_enabled").includes("on"),
    capabilities: {},
    metadata: {},
    costs: [
      {
        unit: (formData.get("unit") as LlmCostUnit) || LlmCostUnit.RUN,
        credit_cost: creditCost,
        credential_provider:
          provider.default_credential_provider || provider.name,
        credential_id: provider.default_credential_id || undefined,
        credential_type: provider.default_credential_type || "api_key",
        metadata: {},
      },
    ],
  };

  const response = await postV2CreateLlmModel(payload);
  if (response.status !== 200) {
    throw new Error("Failed to create LLM model");
  }
  revalidatePath(ADMIN_LLM_PATH);
}

export async function updateLlmModelAction(formData: FormData) {
  const modelId = getRequiredFormField(formData, "model_id", "Model id");
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
    is_enabled: formData.has("is_enabled")
      ? formData.getAll("is_enabled").includes("on")
      : undefined,
    costs: formData.get("credit_cost")
      ? [
          {
            unit: (formData.get("unit") as LlmCostUnit) || LlmCostUnit.RUN,
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

  const response = await patchV2UpdateLlmModel(modelId, payload);
  if (response.status !== 200) {
    throw new Error("Failed to update LLM model");
  }
  revalidatePath(ADMIN_LLM_PATH);
}

export async function toggleLlmModelAction(formData: FormData): Promise<void> {
  const modelId = getRequiredFormField(formData, "model_id", "Model id");
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

  const response = await patchV2ToggleLlmModelAvailability(modelId, payload);
  if (response.status !== 200) {
    throw new Error("Failed to toggle LLM model");
  }
  revalidatePath(ADMIN_LLM_PATH);
}

export async function deleteLlmModelAction(formData: FormData): Promise<void> {
  const modelId = getRequiredFormField(formData, "model_id", "Model id");
  const rawReplacement = formData.get("replacement_model_slug");
  const replacementModelSlug =
    rawReplacement && String(rawReplacement).trim()
      ? String(rawReplacement).trim()
      : undefined;

  const response = await deleteV2DeleteLlmModelAndMigrateWorkflows(modelId, {
    replacement_model_slug: replacementModelSlug,
  });
  if (response.status !== 200) {
    throw new Error("Failed to delete model");
  }
  revalidatePath(ADMIN_LLM_PATH);
}

export async function fetchLlmModelUsage(
  modelId: string,
): Promise<LlmModelUsageResponse> {
  const response = await getV2GetModelUsageCount(modelId);
  if (response.status !== 200) {
    throw new Error("Failed to fetch model usage");
  }
  return response.data;
}

// =============================================================================
// Migration Actions
// =============================================================================

export async function fetchLlmMigrations(
  includeReverted: boolean = false,
): Promise<LlmMigrationsResponse> {
  const response = await getV2ListModelMigrations({
    include_reverted: includeReverted,
  });
  if (response.status !== 200) {
    throw new Error("Failed to fetch migrations");
  }
  return response.data;
}

export async function revertLlmMigrationAction(
  formData: FormData,
): Promise<void> {
  const migrationId = getRequiredFormField(
    formData,
    "migration_id",
    "Migration id",
  );

  const response = await postV2RevertAModelMigration(migrationId, null);
  if (response.status !== 200) {
    throw new Error("Failed to revert migration");
  }
  revalidatePath(ADMIN_LLM_PATH);
}

// =============================================================================
// Creator Actions
// =============================================================================

export async function fetchLlmCreators(): Promise<LlmCreatorsResponse> {
  const response = await getV2ListModelCreators();
  if (response.status !== 200) {
    throw new Error("Failed to fetch creators");
  }
  return response.data;
}

export async function createLlmCreatorAction(
  formData: FormData,
): Promise<void> {
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

  const response = await postV2CreateModelCreator(payload);
  if (response.status !== 200) {
    throw new Error("Failed to create creator");
  }
  revalidatePath(ADMIN_LLM_PATH);
}

export async function updateLlmCreatorAction(
  formData: FormData,
): Promise<void> {
  const creatorId = getRequiredFormField(formData, "creator_id", "Creator id");

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

  const response = await patchV2UpdateModelCreator(creatorId, payload);
  if (response.status !== 200) {
    throw new Error("Failed to update creator");
  }
  revalidatePath(ADMIN_LLM_PATH);
}

export async function deleteLlmCreatorAction(
  formData: FormData,
): Promise<void> {
  const creatorId = getRequiredFormField(formData, "creator_id", "Creator id");

  const response = await deleteV2DeleteModelCreator(creatorId);
  if (response.status !== 200) {
    throw new Error("Failed to delete creator");
  }
  revalidatePath(ADMIN_LLM_PATH);
}

// =============================================================================
// Recommended Model Actions
// =============================================================================

export async function setRecommendedModelAction(
  formData: FormData,
): Promise<void> {
  const modelId = getRequiredFormField(formData, "model_id", "Model id");

  const response = await postV2SetRecommendedModel({ model_id: modelId });
  if (response.status !== 200) {
    throw new Error("Failed to set recommended model");
  }

  revalidatePath(ADMIN_LLM_PATH);
}
