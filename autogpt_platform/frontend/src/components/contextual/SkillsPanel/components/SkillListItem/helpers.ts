import type { CopilotSkillDetail } from "@/app/api/__generated__/models/copilotSkillDetail";
import type { CopilotSkillInfo } from "@/app/api/__generated__/models/copilotSkillInfo";

const DESCRIPTION_PREVIEW_MAX_LEN = 220;

export function describeSkill(skill: CopilotSkillInfo) {
  const description = (skill.description ?? "").trim();
  const descriptionPreview =
    description.length === 0
      ? "(no description)"
      : description.length > DESCRIPTION_PREVIEW_MAX_LEN
        ? `${description.slice(0, DESCRIPTION_PREVIEW_MAX_LEN).trimEnd()}…`
        : description;

  const triggers = (skill.triggers ?? []).filter((t) => t.trim().length > 0);

  return { descriptionPreview, triggers };
}

// Rebuild a canonical, re-uploadable SKILL.md from a fetched skill detail.
// JSON string/array literals are valid YAML, so this round-trips cleanly back
// through the upload endpoint's `parse_skill_markdown` without a YAML lib.
export function renderSkillMarkdown(detail: CopilotSkillDetail): string {
  const frontmatter = [
    "---",
    `name: ${JSON.stringify(detail.name)}`,
    `description: ${JSON.stringify(detail.description)}`,
  ];
  const triggers = (detail.triggers ?? []).filter((t) => t.trim().length > 0);
  if (triggers.length > 0) {
    frontmatter.push(`triggers: ${JSON.stringify(triggers)}`);
  }
  if (detail.version) {
    frontmatter.push(`version: ${JSON.stringify(detail.version)}`);
  }
  frontmatter.push("---");

  return `${frontmatter.join("\n")}\n\n${(detail.body ?? "").trim()}\n`;
}

export function downloadTextFile(filename: string, text: string): void {
  const blob = new Blob([text], { type: "text/markdown;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}
