/**
 * Hook for LLM Registry page data fetching and state management.
 */

import {
  fetchLlmModels,
  fetchLlmProviders,
} from "./actions";

export async function useLlmRegistryPage() {
  const [providersResponse, modelsResponse] = await Promise.all([
    fetchLlmProviders(),
    fetchLlmModels(),
  ]);

  return {
    providers: providersResponse.providers,
    models: modelsResponse.models,
  };
}

