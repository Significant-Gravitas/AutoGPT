import { getGetExpertQueryKey } from "@/app/api/__generated__/endpoints/experts/experts";
import {
  getListSkillLearningHistoryQueryKey,
  useListSkillLearningHistory,
  useUpdateExpertLearningPolicy,
} from "@/app/api/__generated__/endpoints/skill-learning/skill-learning";
import { Expert } from "@/app/api/__generated__/models/expert";
import { LearningHistoryItem } from "@/app/api/__generated__/models/learningHistoryItem";
import { okData } from "@/app/api/helpers";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";

const HISTORY_LIMIT = 50;

export interface OpenSkill {
  name: string;
  versionId: string | null;
}

export function useExpertLearning(
  expert: Expert,
  initialSkill: string | null,
  initialVersionId: string | null,
) {
  const enabled = Boolean(useGetFlag(Flag.DREAM_SKILL_LEARNING_ENABLED));
  const queryClient = useQueryClient();
  const { toast } = useToast();
  const [openSkill, setOpenSkill] = useState<OpenSkill | null>(
    initialSkill ? { name: initialSkill, versionId: initialVersionId } : null,
  );
  // A deliberate navigation (chat chip, history link) names a new record;
  // follow it instead of keeping the previously opened one.
  useEffect(() => {
    if (initialSkill) {
      setOpenSkill({ name: initialSkill, versionId: initialVersionId });
    }
  }, [initialSkill, initialVersionId]);

  const historyQuery = useListSkillLearningHistory(
    { expert_id: expert.id, limit: HISTORY_LIMIT },
    { query: { enabled, select: (res) => okData(res)?.items ?? [] } },
  );
  const items = historyQuery.data ?? [];
  const latestBySkill = new Map<string, LearningHistoryItem>();
  for (const item of items) {
    if (item.kind !== "version" || !item.skill_name) continue;
    if (!latestBySkill.has(item.skill_name.toLowerCase())) {
      latestBySkill.set(item.skill_name.toLowerCase(), item);
    }
  }
  const recentChange =
    items.find((item) => item.kind === "version" && item.state === "ready") ??
    null;

  const { mutate: updatePolicy, isPending: isTogglingLearning } =
    useUpdateExpertLearningPolicy({
      mutation: {
        onSuccess: (_res, variables) => {
          queryClient.invalidateQueries({
            queryKey: getGetExpertQueryKey(expert.id),
          });
          toast({
            title: variables.data.learning_paused
              ? `Learning paused for ${expert.name}`
              : `Learning resumed for ${expert.name}`,
            description: variables.data.learning_paused
              ? "Existing skills stay in use; no new automatic changes."
              : undefined,
          });
        },
        onError: () =>
          toast({ title: "Could not update learning", variant: "destructive" }),
      },
    });

  function setLearningPaused(paused: boolean) {
    updatePolicy({ expertId: expert.id, data: { learning_paused: paused } });
  }

  function learningLineFor(skillName: string) {
    const latest = latestBySkill.get(skillName.toLowerCase());
    if (!latest) return null;
    const parts = [
      latest.version !== null && latest.version !== undefined
        ? `v${latest.version}`
        : null,
      latest.origin_label ?? null,
      latest.state_label,
    ].filter((part): part is string => Boolean(part));
    return {
      label: parts.join(" · "),
      onDetails: () =>
        setOpenSkill({ name: skillName, versionId: latest.version_id ?? null }),
    };
  }

  function refreshHistory() {
    queryClient.invalidateQueries({
      queryKey: getListSkillLearningHistoryQueryKey({
        expert_id: expert.id,
        limit: HISTORY_LIMIT,
      }),
    });
  }

  return {
    enabled,
    isLearningPaused: Boolean(expert.learning_paused_at),
    setLearningPaused,
    isTogglingLearning,
    recentChange,
    learningLineFor,
    openSkill,
    openSkillDetails: (name: string) => setOpenSkill({ name, versionId: null }),
    closeSkill: () => setOpenSkill(null),
    refreshHistory,
  };
}
