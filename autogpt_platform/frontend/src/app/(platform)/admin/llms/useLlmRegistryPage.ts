/**
 * Hook for LLM Registry page data fetching and state management.
 */

import {
  fetchLlmMigrations,
  fetchLlmModels,
  fetchLlmProviders,
} from "./actions";

export async function useLlmRegistryPage() {
  // Fetch providers and models (required)
  const [providersResponse, modelsResponse] = await Promise.all([
    fetchLlmProviders(),
    fetchLlmModels(),
  ]);

  // Fetch migrations separately with fallback (table might not exist yet)
  let migrations: Awaited<ReturnType<typeof fetchLlmMigrations>>["migrations"] =
    [];
  try {
    const migrationsResponse = await fetchLlmMigrations(false);
    migrations = migrationsResponse.migrations;
  } catch {
    // Migrations table might not exist yet - that's ok, just show empty list
    console.warn("Could not fetch migrations - table may not exist yet");
  }

  return {
    providers: providersResponse.providers,
    models: modelsResponse.models,
    migrations,
  };
}

