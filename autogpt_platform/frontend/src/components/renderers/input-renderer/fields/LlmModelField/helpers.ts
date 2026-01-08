import { LlmModelMetadata } from "./types";

const displayNameOverrides: Record<string, string> = {
  aiml_api: "AI/ML",
  anthropic: "Anthropic",
  openai: "OpenAI",
  open_router: "Open Router",
  llama_api: "Llama API",
  groq: "Groq",
  ollama: "Ollama",
  v0: "V0",
};

export function toLlmDisplayName(value: string): string {
  if (!value) {
    return "";
  }
  const normalized = value.toLowerCase();
  if (displayNameOverrides[normalized]) {
    return displayNameOverrides[normalized];
  }
  return value
    .split(/[_-]/)
    .map((word) =>
      word.length ? word[0].toUpperCase() + word.slice(1).toLowerCase() : "",
    )
    .join(" ");
}

export function groupByCreator(models: LlmModelMetadata[]) {
  const map = new Map<string, LlmModelMetadata[]>();
  for (const model of models) {
    const existing = map.get(model.creator) ?? [];
    existing.push(model);
    map.set(model.creator, existing);
  }
  return map;
}

export function groupByTitle(models: LlmModelMetadata[]) {
  const map = new Map<string, LlmModelMetadata[]>();
  for (const model of models) {
    const existing = map.get(model.title) ?? [];
    existing.push(model);
    map.set(model.title, existing);
  }
  return map;
}
