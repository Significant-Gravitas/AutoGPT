import {
  fetchProviders,
  fetchModels,
  fetchCreators,
  fetchMigrations,
} from "./actions";

export async function getLlmRegistryPageData() {
  // Fetch all data in parallel
  const [providersData, modelsData, creatorsData, migrationsData] =
    await Promise.all([
      fetchProviders(),
      fetchModels(),
      fetchCreators(),
      fetchMigrations(),
    ]);

  return {
    providers: providersData.providers || [],
    models: modelsData.models || [],
    creators: creatorsData.creators || [],
    migrations: migrationsData.migrations || [],
  };
}
