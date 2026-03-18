"use server";

import { revalidatePath } from "next/cache";

const ADMIN_LLM_PATH = "/admin/llms";
const API_BASE = process.env.NEXT_PUBLIC_AGPT_SERVER_URL || "http://localhost:8000";

// Helper to make authenticated API calls
async function apiCall(
  endpoint: string,
  options: RequestInit = {}
): Promise<Response> {
  // TODO: Add auth token from session
  const headers = new Headers(options.headers);
  headers.set("Content-Type", "application/json");
  
  const response = await fetch(`${API_BASE}${endpoint}`, {
    ...options,
    headers,
  });

  if (!response.ok) {
    const error = await response.text();
    throw new Error(`API Error (${response.status}): ${error}`);
  }

  return response;
}

// =============================================================================
// Provider Actions
// =============================================================================

export async function createProvider(formData: FormData) {
  try {
    const data = {
      name: formData.get("name") as string,
      display_name: formData.get("display_name") as string,
      description: formData.get("description") as string || null,
      default_credential_provider: formData.get("default_credential_provider") as string || null,
      metadata: {},
    };

    await apiCall("/api/llm/providers", {
      method: "POST",
      body: JSON.stringify(data),
    });

    revalidatePath(ADMIN_LLM_PATH);
    return { success: true };
  } catch (error) {
    console.error("Failed to create provider:", error);
    return { success: false, error: String(error) };
  }
}

export async function updateProvider(name: string, formData: FormData) {
  try {
    const data = {
      display_name: formData.get("display_name") as string,
      description: formData.get("description") as string || null,
      default_credential_provider: formData.get("default_credential_provider") as string || null,
      metadata: {},
    };

    await apiCall(`/api/llm/providers/${name}`, {
      method: "PATCH",
      body: JSON.stringify(data),
    });

    revalidatePath(ADMIN_LLM_PATH);
    return { success: true };
  } catch (error) {
    console.error("Failed to update provider:", error);
    return { success: false, error: String(error) };
  }
}

export async function deleteProvider(name: string) {
  try {
    await apiCall(`/api/llm/providers/${name}`, {
      method: "DELETE",
    });

    revalidatePath(ADMIN_LLM_PATH);
    return { success: true };
  } catch (error) {
    console.error("Failed to delete provider:", error);
    return { success: false, error: String(error) };
  }
}

// =============================================================================
// Model Actions
// =============================================================================

export async function createModel(formData: FormData) {
  try {
    const data = {
      slug: formData.get("slug") as string,
      display_name: formData.get("display_name") as string,
      description: formData.get("description") as string || null,
      provider_id: formData.get("provider_id") as string,
      creator_id: formData.get("creator_id") as string || null,
      context_window: parseInt(formData.get("context_window") as string),
      max_output_tokens: formData.get("max_output_tokens") 
        ? parseInt(formData.get("max_output_tokens") as string) 
        : null,
      price_tier: parseInt(formData.get("price_tier") as string),
      is_enabled: formData.get("is_enabled") === "true",
      is_recommended: formData.get("is_recommended") === "true",
      supports_tools: formData.get("supports_tools") === "true",
      supports_json_output: formData.get("supports_json_output") === "true",
      supports_reasoning: formData.get("supports_reasoning") === "true",
      supports_parallel_tool_calls: formData.get("supports_parallel_tool_calls") === "true",
      capabilities: {},
      metadata: {},
    };

    await apiCall("/api/llm/models", {
      method: "POST",
      body: JSON.stringify(data),
    });

    revalidatePath(ADMIN_LLM_PATH);
    return { success: true };
  } catch (error) {
    console.error("Failed to create model:", error);
    return { success: false, error: String(error) };
  }
}

export async function updateModel(slug: string, formData: FormData) {
  try {
    const data: Record<string, any> = {};
    
    // Only include fields that are present in formData
    const displayName = formData.get("display_name");
    if (displayName) data.display_name = displayName as string;
    
    const description = formData.get("description");
    if (description !== null) data.description = description as string;
    
    const contextWindow = formData.get("context_window");
    if (contextWindow) data.context_window = parseInt(contextWindow as string);
    
    const maxOutputTokens = formData.get("max_output_tokens");
    if (maxOutputTokens) data.max_output_tokens = parseInt(maxOutputTokens as string);
    
    const priceTier = formData.get("price_tier");
    if (priceTier) data.price_tier = parseInt(priceTier as string);
    
    const isEnabled = formData.get("is_enabled");
    if (isEnabled !== null) data.is_enabled = isEnabled === "true";
    
    const isRecommended = formData.get("is_recommended");
    if (isRecommended !== null) data.is_recommended = isRecommended === "true";

    await apiCall(`/api/llm/models/${slug}`, {
      method: "PATCH",
      body: JSON.stringify(data),
    });

    revalidatePath(ADMIN_LLM_PATH);
    return { success: true };
  } catch (error) {
    console.error("Failed to update model:", error);
    return { success: false, error: String(error) };
  }
}

export async function deleteModel(slug: string) {
  try {
    await apiCall(`/api/llm/models/${slug}`, {
      method: "DELETE",
    });

    revalidatePath(ADMIN_LLM_PATH);
    return { success: true };
  } catch (error) {
    console.error("Failed to delete model:", error);
    return { success: false, error: String(error) };
  }
}

// =============================================================================
// Data Fetching (for page load)
// =============================================================================

export async function fetchProviders() {
  try {
    const response = await apiCall("/api/llm/providers");
    return await response.json();
  } catch (error) {
    console.error("Failed to fetch providers:", error);
    return { providers: [] };
  }
}

export async function fetchModels() {
  try {
    const response = await apiCall("/api/llm/models");
    return await response.json();
  } catch (error) {
    console.error("Failed to fetch models:", error);
    return { models: [], total: 0 };
  }
}

// Placeholder for features not yet implemented in backend
export async function fetchCreators() {
  return { creators: [] };
}

export async function fetchMigrations() {
  return { migrations: [] };
}
