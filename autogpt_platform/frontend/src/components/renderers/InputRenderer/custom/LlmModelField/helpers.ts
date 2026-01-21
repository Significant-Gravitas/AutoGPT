import { LlmModelMetadata } from "./types";

export function groupByCreator(models: LlmModelMetadata[]) {
  const map = new Map<string, LlmModelMetadata[]>();
  for (const model of models) {
    const key = getCreatorDisplayName(model);
    const existing = map.get(key) ?? [];
    existing.push(model);
    map.set(key, existing);
  }
  return map;
}

export function groupByTitle(models: LlmModelMetadata[]) {
  const map = new Map<string, LlmModelMetadata[]>();
  for (const model of models) {
    const displayName = getModelDisplayName(model);
    const existing = map.get(displayName) ?? [];
    existing.push(model);
    map.set(displayName, existing);
  }
  return map;
}

export function getCreatorDisplayName(model: LlmModelMetadata): string {
  return model.creator_name || model.creator || "";
}

export function getModelDisplayName(model: LlmModelMetadata): string {
  return model.title || model.name || "";
}

export function getProviderDisplayName(model: LlmModelMetadata): string {
  return model.provider_name || model.provider || "";
}
