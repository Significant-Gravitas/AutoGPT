import type { SkillVersionSummary } from "@/app/api/__generated__/models/skillVersionSummary";

export const CHAT_SOURCE_KIND = "chat_session";

/**
 * Where a skill's learning detail opens: the Expert's Skills tab, or the
 * Memory page for the personal (AutoPilot) scope. Both open the exact
 * version when one is named.
 */
export function skillDetailHref(args: {
  expertId: string | null | undefined;
  skillName: string;
  versionId?: string | null;
}) {
  const params = new URLSearchParams();
  if (args.expertId) params.set("tab", "skills");
  params.set("skill", args.skillName);
  if (args.versionId) params.set("version", args.versionId);
  if (!args.expertId) return `/settings/memory?${params.toString()}`;
  return `/team/${args.expertId}?${params.toString()}`;
}

/** Restorable: not current, has content, and not invalidated or blocked. */
export function canRestore(
  version: SkillVersionSummary,
  currentVersionId: string | null,
) {
  return (
    version.id !== currentVersionId &&
    Boolean(version.body) &&
    version.state !== "invalidated" &&
    version.state !== "blocked_content"
  );
}

export function versionLabel(version: SkillVersionSummary) {
  return `v${version.version} · ${version.origin_label}`;
}

export function evidenceLine(version: SkillVersionSummary) {
  const primary = (version.evidence ?? []).find(
    (item) => item.kind === "outcome",
  );
  return primary?.label ?? reuseLabel(version);
}

export function reuseLabel(version: SkillVersionSummary) {
  return version.use?.reuse_label ?? "Not yet reused";
}

export function recentChangeLine(version: SkillVersionSummary) {
  const verb =
    version.origin === "restored"
      ? "Restored"
      : version.version === 1
        ? "Added"
        : "Updated";
  return `${verb} ${version.skill_name} · ${version.origin_label} · ${evidenceLine(version)}`;
}

export interface StepChange {
  step: string;
  before: string | null;
  after: string | null;
}

/**
 * Before/after grouped by procedure step (numbered or bulleted lines). Steps
 * absent on one side render as added/removed; prose outside lists is
 * compared as a single "Other text" entry.
 */
export function stepChanges(before: string, after: string): StepChange[] {
  const previous = stepMap(before);
  const next = stepMap(after);
  const keys = Array.from(new Set([...previous.keys(), ...next.keys()]));
  return keys
    .map((key) => ({
      step: key,
      before: previous.get(key) ?? null,
      after: next.get(key) ?? null,
    }))
    .filter((change) => change.before !== change.after);
}

function stepMap(body: string) {
  const steps = new Map<string, string>();
  const prose: string[] = [];
  let index = 0;
  for (const raw of stripFrontmatter(body).split("\n")) {
    const match = raw.match(/^\s*(?:(\d+)[.)]|[-*•])\s+(.+)$/);
    if (match) {
      index += 1;
      steps.set(`Step ${match[1] ?? index}`, match[2].trim());
    } else if (raw.trim() && !raw.trim().startsWith("#")) {
      prose.push(raw.trim());
    }
  }
  if (prose.length) steps.set("Other text", prose.join(" "));
  return steps;
}

export function stripFrontmatter(content: string) {
  const match = content.match(/^---\n[\s\S]*?\n---\n?/);
  return match ? content.slice(match[0].length) : content;
}

export function lineDiff(before: string, after: string) {
  const previous = new Set(stripFrontmatter(before).split("\n"));
  const next = new Set(stripFrontmatter(after).split("\n"));
  const removed = [...previous].filter(
    (line) => !next.has(line) && line.trim(),
  );
  const added = [...next].filter((line) => !previous.has(line) && line.trim());
  return [
    ...removed.map((line) => `- ${line}`),
    ...added.map((line) => `+ ${line}`),
  ].join("\n");
}
