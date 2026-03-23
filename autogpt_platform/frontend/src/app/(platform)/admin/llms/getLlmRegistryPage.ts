import {
  fetchLlmProviders,
  fetchLlmModels,
  fetchLlmCreators,
  fetchLlmMigrations,
} from "./actions";

export async function getLlmRegistryPageData() {
  // Fetch all data in parallel
  const [providersData, modelsData, creatorsData, migrationsData] =
    await Promise.all([
      fetchLlmProviders(),
      fetchLlmModels(),
      fetchLlmCreators(),
      fetchLlmMigrations(),
    ]);

  return {
    providers: providersData.providers || [],
    models: modelsData.models || [],
    creators: creatorsData.creators || [],
    migrations: migrationsData.migrations || [],
  };
}
