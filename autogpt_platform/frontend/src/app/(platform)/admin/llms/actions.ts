"use server";

import BackendApi from "@/lib/autogpt-server-api";
import type {
  CreateLlmModelRequest,
  LlmModelsResponse,
  LlmProvidersResponse,
  ToggleLlmModelRequest,
  UpdateLlmModelRequest,
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
      ? String(formData.get("default_credential_provider"))
      : undefined,
    default_credential_id: formData.get("default_credential_id")
      ? String(formData.get("default_credential_id"))
      : undefined,
    default_credential_type: formData.get("default_credential_type")
      ? String(formData.get("default_credential_type"))
      : undefined,
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
  const payload: CreateLlmModelRequest = {
    slug: String(formData.get("slug") || "").trim(),
    display_name: String(formData.get("display_name") || "").trim(),
    description: formData.get("description")
      ? String(formData.get("description"))
      : undefined,
    provider_id: String(formData.get("provider_id")),
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
    ],
  };

  const api = new BackendApi();
  await api.createAdminLlmModel(payload);
  revalidatePath(ADMIN_LLM_PATH);
}

export async function updateLlmModelAction(formData: FormData) {
  const modelId = String(formData.get("model_id"));
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

export async function toggleLlmModelAction(formData: FormData) {
  const modelId = String(formData.get("model_id"));
  const shouldEnable = formData.get("is_enabled") === "true";
  const payload: ToggleLlmModelRequest = {
    is_enabled: shouldEnable,
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

