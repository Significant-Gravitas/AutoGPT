import type { CopilotSkillDetail } from "@/app/api/__generated__/models/copilotSkillDetail";
import type { CopilotSkillFile } from "@/app/api/__generated__/models/copilotSkillFile";
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

export interface SkillFileRow {
  path: string;
  depth: number;
  label: string;
  sizeLabel: string | null;
}

// Flatten a package's relative paths into indented rows, each directory once
// above its files, so the dialog renders a tree without recursing.
export function buildSkillFileRows(
  files: CopilotSkillFile[] | undefined,
): SkillFileRow[] {
  const rows: SkillFileRow[] = [];
  const seen = new Set<string>();
  const sorted = [...(files ?? [])].sort((a, b) =>
    a.path.localeCompare(b.path),
  );

  for (const file of sorted) {
    const segments = file.path.split("/");
    segments.forEach((segment, index) => {
      const path = segments.slice(0, index + 1).join("/");
      if (seen.has(path)) return;
      seen.add(path);
      const isLeaf = index === segments.length - 1;
      rows.push({
        path,
        depth: index,
        label: isLeaf ? segment : `${segment}/`,
        sizeLabel: isLeaf ? formatFileSize(file.size_bytes) : null,
      });
    });
  }
  return rows;
}

function formatFileSize(bytes: number): string {
  if (bytes >= 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${bytes} B`;
}

export function downloadFile(filename: string, blob: Blob): void {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}
