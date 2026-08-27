const INTERNAL_BLOCK =
  /<(?:user_context|memory_context|env_context|budget_context|session_context|available_skills|expert_identity|expert_workflows|team_context)\b[^>]*>[\s\S]*?<\/(?:user_context|memory_context|env_context|budget_context|session_context|available_skills|expert_identity|expert_workflows|team_context)>/gi;
const CODE_FENCE = /```[\s\S]*?```/g;
const ABSOLUTE_PATH =
  /(?:[a-z]:\\(?:users|temp|windows|workspace)\\|\/(?:Users|home|tmp|private|var|opt|workspace|mnt)\/)[^\s,;)]+/gi;
const UUID =
  /\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b/gi;
const INTERNAL_ID =
  /\b(?:tool(?:_call)?|graph|block|session|execution|node(?:_exec)?)[_-]?id\s*[:=]\s*["']?[a-z0-9._:-]+["']?/gi;
const INLINE_JSON = /\{[^{}\n]{1,2000}\}/g;

export function founderSafeText(value: string, fallback: string) {
  const trimmed = value.trim();
  if (
    (trimmed.startsWith("{") && trimmed.endsWith("}")) ||
    (trimmed.startsWith("[") && trimmed.endsWith("]"))
  ) {
    return fallback;
  }

  const safe = trimmed
    .replace(INTERNAL_BLOCK, "")
    .replace(CODE_FENCE, "")
    .replace(INLINE_JSON, "")
    .replace(ABSOLUTE_PATH, "a workspace file")
    .replace(INTERNAL_ID, "internal reference")
    .replace(UUID, "reference")
    .split("\n")
    .filter(
      (line) =>
        !/^\s*(?:\$ |stdout\s*:|stderr\s*:|exit\s+code\s*:)/i.test(line),
    )
    .join(" ")
    .replace(/\s+/g, " ")
    .trim();

  return safe || fallback;
}

export function founderSafeArtifactName(value: string) {
  const name = value.split(/[\\/]/).at(-1)?.trim() ?? "";
  return founderSafeText(name, "Deliverable");
}
