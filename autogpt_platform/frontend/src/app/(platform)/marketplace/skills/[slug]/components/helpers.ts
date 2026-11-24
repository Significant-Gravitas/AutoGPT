import type { CredentialsMetaResponse } from "@/app/api/__generated__/models/credentialsMetaResponse";

/** Every name a skill's `required_providers` can use for what the user has
 *  connected. An MCP service goes by its catalog entry ("mcp_linear"), which
 *  the backend reads off the sign-in's server; its provider is only "mcp". */
export function connectedProviders(
  credentials: CredentialsMetaResponse[],
): Set<string> {
  return new Set(
    credentials.flatMap((credential) =>
      credential.mcp_catalog_entry
        ? [credential.provider, credential.mcp_catalog_entry]
        : [credential.provider],
    ),
  );
}

const HEADING = /^(#{1,6})\s+(.*)$/;
const FENCE = /^\s*(```|~~~)/;

/** Strip the body's own leading title when it repeats the page's, then shift
 *  every heading so the shallowest lands on `<h3>` — the page owns `<h1>` and
 *  each section owns `<h2>`, and a `##`-only body should not start at `<h4>`.
 *  A leading `# ` that says something else is kept. */
export function prepareSkillBody(body: string, title: string): string {
  const lines = body.split("\n");
  const headings = collectHeadings(lines);

  const first = headings[0];
  const isLeadingTitle =
    first !== undefined &&
    first.level === 1 &&
    lines.slice(0, first.index).every((line) => line.trim() === "") &&
    first.text.trim().toLowerCase() === title.trim().toLowerCase();

  const kept = isLeadingTitle ? headings.slice(1) : headings;
  if (kept.length === 0)
    return isLeadingTitle ? dropLine(lines, first.index) : body;

  const shallowest = Math.min(...kept.map((heading) => heading.level));
  const shift = 3 - shallowest;

  const shifted = lines.map((line, index) => {
    const heading = kept.find((entry) => entry.index === index);
    if (!heading) return line;
    const level = Math.min(6, Math.max(1, heading.level + shift));
    return `${"#".repeat(level)} ${heading.text}`;
  });

  return isLeadingTitle ? dropLine(shifted, first.index) : shifted.join("\n");
}

function collectHeadings(lines: string[]) {
  const headings: Array<{ index: number; level: number; text: string }> = [];
  let inFence = false;
  lines.forEach((line, index) => {
    if (FENCE.test(line)) {
      inFence = !inFence;
      return;
    }
    if (inFence) return;
    const match = HEADING.exec(line);
    if (match) {
      headings.push({ index, level: match[1].length, text: match[2] });
    }
  });
  return headings;
}

function dropLine(lines: string[], index: number): string {
  return lines
    .filter((_, position) => position !== index)
    .join("\n")
    .replace(/^\n+/, "");
}
