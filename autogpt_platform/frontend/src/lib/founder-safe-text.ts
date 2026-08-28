const INTERNAL_BLOCK =
  /<(?:user_context|memory_context|env_context|budget_context|session_context|available_skills|expert_identity|expert_workflows|team_context|project_context)\b[^>]*>[\s\S]*?<\/(?:user_context|memory_context|env_context|budget_context|session_context|available_skills|expert_identity|expert_workflows|team_context|project_context)>/gi;
const CODE_FENCE = /```[\s\S]*?```/g;
const HTML_COMMENT = /<!--[\s\S]*?-->/g;
const ABSOLUTE_PATH =
  /(?:[a-z]:\\(?:users|temp|windows|workspace)\\|\/(?:Users|home|root|tmp|private|var|opt|workspace|documents|mnt)\/)[^\s,;)]+/gi;
const UUID =
  /\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b/gi;
const INTERNAL_ID =
  /\b(?:tool(?:_call)?|graph|block|session|execution|node(?:_exec)?)[_-]?id\s*[:=]\s*["']?[a-z0-9._:-]+["']?/gi;
const INLINE_JSON = /\{[^{}\n]{1,2000}\}/g;
const WORKSPACE_URI = /workspace:\/\/[^\s)\]]+/g;
const INTERNAL_ERROR_CODE =
  /\b(?:DELEGATION_PERSISTENCE_FAILED|[A-Z][A-Z0-9]+(?:_[A-Z0-9]+){2,})\b/g;
const INFRASTRUCTURE_ERROR =
  /(?:client is not connected to the query engine|prisma(?: client)?|query engine)/gi;
const INTERNAL_BLOCK_NAME = /\b[A-Z][A-Za-z0-9]*(?:Generator)?Block\b/g;

function sanitizeFounderText(
  value: string,
  fallback: string,
  separator: string,
) {
  const workspaceUris: string[] = [];
  const trimmed = value
    .replace(WORKSPACE_URI, (uri) => {
      const index = workspaceUris.push(uri) - 1;
      return `__FOUNDER_WORKSPACE_${index}__`;
    })
    .trim();
  if (
    (trimmed.startsWith("{") && trimmed.endsWith("}")) ||
    (trimmed.startsWith("[") && trimmed.endsWith("]"))
  ) {
    return fallback;
  }

  const safe = trimmed
    .replace(INTERNAL_BLOCK, "")
    .replace(CODE_FENCE, "")
    .replace(HTML_COMMENT, "")
    .replace(INLINE_JSON, "")
    .replace(ABSOLUTE_PATH, "a workspace file")
    .replace(INTERNAL_ID, "internal reference")
    .replace(UUID, "reference")
    .replace(INTERNAL_ERROR_CODE, "internal service issue")
    .replace(INFRASTRUCTURE_ERROR, "the assignment service is unavailable")
    .replace(INTERNAL_BLOCK_NAME, "workflow step")
    .split("\n")
    .filter(
      (line) =>
        !/^\s*(?:\$ |stdout\s*:|stderr\s*:|exit\s+code\s*:)/i.test(line),
    )
    .join(separator)
    .replace(separator === " " ? /\s+/g : /[\t ]+/g, " ")
    .replace(/\n{3,}/g, "\n\n")
    .trim();

  const restored = safe.replace(
    /__FOUNDER_WORKSPACE_(\d+)__/g,
    (_, index: string) => workspaceUris[Number(index)] ?? "",
  );
  return restored || fallback;
}

export function founderSafeText(value: string, fallback: string) {
  return sanitizeFounderText(value, fallback, " ");
}

export function founderSafeMarkdown(value: string, fallback: string) {
  return sanitizeFounderText(value, fallback, "\n");
}

export function founderSafeArtifactName(value: string) {
  const name = value.split(/[\\/]/).at(-1)?.trim() ?? "";
  return founderSafeText(name, "Deliverable");
}
