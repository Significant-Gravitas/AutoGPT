import type { LlmModelMetadata } from "./types";

type ModelEntry = {
  title: string;
  entries: Pick<LlmModelMetadata, "provider" | "name">[];
};

export function compareModelEntries(a: ModelEntry, b: ModelEntry) {
  if (
    a.entries.every((model) => model.provider === "minimax") &&
    b.entries.every((model) => model.provider === "minimax")
  ) {
    const aFlash = a.entries.some(
      (model) => model.name === "MiniMax-M3.1-Flash",
    );
    const bFlash = b.entries.some(
      (model) => model.name === "MiniMax-M3.1-Flash",
    );
    if (aFlash !== bFlash) return aFlash ? -1 : 1;
  }
  return a.title.localeCompare(b.title);
}

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
