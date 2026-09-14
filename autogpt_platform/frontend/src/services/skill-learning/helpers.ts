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
  const counts = new Map<string, number>();
  let section = "";
  for (const raw of stripFrontmatter(body).split("\n")) {
    const line = raw.trim();
    const heading = line.match(/^#{1,6}\s+(\S.*)$/);
    const match = line.match(/^(?:\d+[.)]|[-*•])\s+(\S.*)$/);
    if (heading) {
      section = heading[1];
      prose.push(line);
    } else if (match) {
      const index = (counts.get(section) ?? 0) + 1;
      counts.set(section, index);
      const prefix =
        !section || section.toLowerCase() === "steps" ? "" : `${section} · `;
      steps.set(`${prefix}Step ${index}`, match[1]);
    } else if (line) {
      prose.push(line);
    }
  }
  if (prose.length) steps.set("Other text", prose.join("\n"));
  return steps;
}

export function stripFrontmatter(content: string) {
  const match = content.match(/^---\n[\s\S]*?\n---\n?/);
  return match ? content.slice(match[0].length) : content;
}

export function lineDiff(before: string, after: string) {
  const previous = stripFrontmatter(before).split("\n");
  const next = stripFrontmatter(after).split("\n");
  let start = 0;
  while (
    start < previous.length &&
    start < next.length &&
    previous[start] === next[start]
  ) {
    start += 1;
  }
  let previousEnd = previous.length;
  let nextEnd = next.length;
  while (
    previousEnd > start &&
    nextEnd > start &&
    previous[previousEnd - 1] === next[nextEnd - 1]
  ) {
    previousEnd -= 1;
    nextEnd -= 1;
  }
  return [
    ...previous.slice(start, previousEnd).map((line) => `- ${line}`),
    ...next.slice(start, nextEnd).map((line) => `+ ${line}`),
  ].join("\n");
}
