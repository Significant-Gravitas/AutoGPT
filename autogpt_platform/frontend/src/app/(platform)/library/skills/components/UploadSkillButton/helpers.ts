// Mirrors backend limits in backend/copilot/tools/skills.py — kept in sync
// manually. These bound the per-turn `<available_skills>` index the copilot
// sees, so the cap is deliberate (see backend MAX_DESCRIPTION_CHARS).
export const MAX_SKILL_DESCRIPTION_CHARS = 200;

const FRONTMATTER_RE = /^---\n([\s\S]*?)\n---/;

// Best-effort, dependency-free pre-flight for an uploaded SKILL.md. The
// backend remains the source of truth — this only catches the common,
// confidently-detectable cases (missing frontmatter, over-long single-line
// description) so the user gets an instant, specific message instead of a
// round-trip. Anything we can't parse confidently (e.g. YAML block scalars)
// returns `null` and falls through to the server's validation.
export function getSkillUploadError(content: string): string | null {
  const match = content.match(FRONTMATTER_RE);
  if (!match) {
    return "File is not a valid SKILL.md — expected YAML frontmatter (---) with a name and description.";
  }

  const frontmatter = match[1];
  const description = readScalarField(frontmatter, "description");
  const name = readScalarField(frontmatter, "name");

  // `parse_skill_markdown` rejects when name or description is missing. Only
  // assert this when we can see plain scalar lines — a block scalar would
  // read as absent here, so skip rather than false-reject.
  if (name === "" || description === "") {
    return "File is not a valid SKILL.md — frontmatter must include a non-empty name and description.";
  }

  if (description && description.length > MAX_SKILL_DESCRIPTION_CHARS) {
    const over = description.length - MAX_SKILL_DESCRIPTION_CHARS;
    return `Description is ${description.length}/${MAX_SKILL_DESCRIPTION_CHARS} characters — trim at least ${over}. It appears in the copilot's per-turn skills index, so it's kept short.`;
  }

  return null;
}

// Reads a single-line scalar value for `key` from a YAML frontmatter block.
// Returns the trimmed (and unquoted) value, "" when the key is present but
// empty, or `undefined` when the key is absent or uses a block scalar
// (`>`/`|`) we can't measure confidently.
function readScalarField(frontmatter: string, key: string): string | undefined {
  const line = frontmatter.match(new RegExp(`^${key}:[ \\t]*(.*)$`, "m"));
  if (!line) return undefined;

  const raw = line[1].trim();
  if (raw === "") return "";
  // Block scalars span multiple lines — not measurable from one line.
  if (
    raw === ">" ||
    raw === "|" ||
    raw.startsWith(">") ||
    raw.startsWith("|")
  ) {
    return undefined;
  }

  // Double-quoted YAML escapes inner quotes as `\"` (this is what
  // `renderSkillMarkdown`'s `JSON.stringify` emits on download). Decode with
  // `JSON.parse` so the measured length matches the backend's parsed value
  // instead of counting the backslash escapes.
  if (raw.startsWith('"') && raw.endsWith('"') && raw.length >= 2) {
    try {
      return JSON.parse(raw);
    } catch {
      return raw.slice(1, -1);
    }
  }
  // Single-quoted YAML escapes a quote by doubling it (`''`).
  if (raw.startsWith("'") && raw.endsWith("'") && raw.length >= 2) {
    return raw.slice(1, -1).replace(/''/g, "'");
  }
  return raw;
}
