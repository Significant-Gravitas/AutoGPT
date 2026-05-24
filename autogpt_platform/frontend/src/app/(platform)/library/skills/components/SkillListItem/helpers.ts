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
