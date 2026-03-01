/**
 * Server-side data fetching for LLM Registry page.
 */

import {
  fetchLlmCreators,
  fetchLlmMigrations,
  fetchLlmModels,
  fetchLlmProviders,
} from "./actions";

export async function getLlmRegistryPageData() {
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

  // Fetch creators separately with fallback (table might not exist yet)
  let creators: Awaited<ReturnType<typeof fetchLlmCreators>>["creators"] = [];
  try {
    const creatorsResponse = await fetchLlmCreators();
    creators = creatorsResponse.creators;
  } catch {
    // Creators table might not exist yet - that's ok, just show empty list
    console.warn("Could not fetch creators - table may not exist yet");
  }

  return {
    providers: providersResponse.providers,
    models: modelsResponse.models,
    migrations,
    creators,
  };
}
