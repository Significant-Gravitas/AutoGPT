import { useListCopilotSkills } from "@/app/api/__generated__/endpoints/skills/skills";
import { okData } from "@/app/api/helpers";
import type { SkillCommand } from "./helpers";

/**
 * The chat's skills that can run as `/name` commands: the expert's own
 * skills in an expert chat, personal Otto's otherwise. Skills that set
 * `user-invocable: false` are left out. `enabled` stays off until the user
 * first types a leading "/", so a chat that never uses one never asks.
 */
export function useSkillCommands(
  expertId: string | null | undefined,
  enabled: boolean,
) {
  const query = useListCopilotSkills(
    expertId ? { expert_id: expertId } : undefined,
    {
      query: {
        select: (res) => okData(res) ?? [],
        enabled,
      },
    },
  );

  const skills: SkillCommand[] = (query.data ?? [])
    .filter((skill) => skill.user_invocable !== false)
    .map((skill) => ({
      name: skill.name,
      description: skill.description,
      argumentHint: skill.argument_hint ?? null,
    }));

  return {
    skills,
    isLoading: enabled && query.isLoading,
  };
}
