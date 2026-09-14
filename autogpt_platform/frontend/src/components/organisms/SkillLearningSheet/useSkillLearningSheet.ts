import {
  getGetSkillLearningDetailQueryKey,
  useDecideSkillLearningProposal,
  useEditLearnedSkill,
  useExcludeSkillLearningSource,
  useGetSkillLearningDetail,
  useReportSkillOutcome,
  useRestoreLearnedSkillVersion,
  useUpdateSkillLearningPolicy,
} from "@/app/api/__generated__/endpoints/skill-learning/skill-learning";
import { getListCopilotSkillsQueryKey } from "@/app/api/__generated__/endpoints/skills/skills";
import { DecisionRequestAction } from "@/app/api/__generated__/models/decisionRequestAction";
import { LearningActionResult } from "@/app/api/__generated__/models/learningActionResult";
import { OutcomeReportRequestOutcome } from "@/app/api/__generated__/models/outcomeReportRequestOutcome";
import { SkillPolicyRequest } from "@/app/api/__generated__/models/skillPolicyRequest";
import { okData } from "@/app/api/helpers";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { ApiError } from "@/lib/autogpt-server-api/helpers";
import { useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

interface Args {
  expertId: string | null;
  skillName: string;
  initialVersionId: string | null;
  onChanged: () => void;
}

export type SheetView = "summary" | "changes" | "sources" | "history";

const SETTLED_STATUSES = new Set(["applied", "recorded"]);

/**
 * Personal scope omits ``expert_id`` entirely; an empty id is never sent.
 */
export function scopeParams(expertId: string | null) {
  return expertId ? { expert_id: expertId } : undefined;
}

export function useSkillLearningSheet({
  expertId,
  skillName,
  initialVersionId,
  onChanged,
}: Args) {
  const queryClient = useQueryClient();
  const { toast } = useToast();
  const [view, setView] = useState<SheetView>("summary");
  const [selectedVersionId, setSelectedVersionId] = useState<string | null>(
    initialVersionId,
  );
  const [editDraft, setEditDraft] = useState<string | null>(null);
  // The current version's id captured when editing began. The server
  // checks it inside its write lock, so a change that lands while the
  // owner types (overnight, or from another tab) becomes a conflict with
  // the draft kept, never a silent overwrite.
  const [editBaseVersionId, setEditBaseVersionId] = useState<string | null>(
    null,
  );
  const [decisionDraft, setDecisionDraft] = useState("");

  const params = scopeParams(expertId);
  const detailQuery = useGetSkillLearningDetail(skillName, params, {
    query: { select: (res) => okData(res) ?? null },
  });
  const detail = detailQuery.data ?? null;
  const selectedVersion =
    (detail?.versions ?? []).find(
      (version) => version.id === selectedVersionId,
    ) ??
    detail?.current_version ??
    null;

  function refresh() {
    queryClient.invalidateQueries({
      queryKey: getGetSkillLearningDetailQueryKey(skillName, params),
    });
    queryClient.invalidateQueries({
      queryKey: getListCopilotSkillsQueryKey(params),
    });
    onChanged();
  }

  /** A 200 envelope can still carry a domain rejection (blocked content,
   *  conflict, stale eligibility). Only a settled outcome clears drafts. */
  function report(result: LearningActionResult, successTitle: string) {
    const settled = SETTLED_STATUSES.has(result.status);
    if (settled) {
      toast({ title: successTitle, variant: "success" });
    } else {
      toast({
        title: result.status_label,
        description: result.reason || "Your draft was kept; adjust and retry.",
        variant: "destructive",
      });
    }
    refresh();
    return settled;
  }

  function fail(title: string) {
    return (error: unknown) =>
      toast({
        title,
        description: error instanceof ApiError ? error.message : undefined,
        variant: "destructive",
      });
  }

  const policy = useUpdateSkillLearningPolicy({
    mutation: {
      onSuccess: refresh,
      onError: fail("Could not update the skill"),
    },
  });
  const restore = useRestoreLearnedSkillVersion({
    mutation: {
      onSuccess: (res) => {
        const data = okData(res);
        if (data) report(data, "Version restored for future use");
      },
      onError: fail("Could not restore that version"),
    },
  });
  const decide = useDecideSkillLearningProposal({
    mutation: {
      onSuccess: (res) => {
        const data = okData(res);
        if (data && report(data, "Decision recorded")) setDecisionDraft("");
      },
      onError: fail("Could not record the decision"),
    },
  });
  const outcome = useReportSkillOutcome({
    mutation: {
      onSuccess: () => {
        toast({ title: "Outcome recorded as evidence for a later review" });
        refresh();
      },
      onError: fail("Could not record the outcome"),
    },
  });
  const edit = useEditLearnedSkill({
    mutation: {
      onSuccess: (res) => {
        const data = okData(res);
        if (data && report(data, "Edit applied")) {
          setEditDraft(null);
          setEditBaseVersionId(null);
        }
      },
      onError: fail("Edit blocked"),
    },
  });
  const exclude = useExcludeSkillLearningSource({
    mutation: {
      onSuccess: () => {
        toast({ title: "Source excluded from learning" });
        refresh();
      },
      onError: fail("Could not exclude that source"),
    },
  });

  return {
    view,
    setView,
    detail,
    isLoading: detailQuery.isLoading,
    isError: detailQuery.isError,
    refetch: detailQuery.refetch,
    selectedVersion,
    selectVersion: setSelectedVersionId,
    editDraft,
    setEditDraft,
    editBaseVersionId,
    startEdit: (body: string, baseVersionId: string | null) => {
      setEditBaseVersionId(baseVersionId);
      setEditDraft(body);
    },
    cancelEdit: () => {
      setEditDraft(null);
      setEditBaseVersionId(null);
    },
    decisionDraft,
    setDecisionDraft,
    isBusy:
      policy.isPending ||
      restore.isPending ||
      decide.isPending ||
      outcome.isPending ||
      edit.isPending ||
      exclude.isPending,
    updatePolicy: (data: SkillPolicyRequest) =>
      policy.mutate({ name: skillName, data, params }),
    restoreVersion: (versionId: string) =>
      restore.mutate({
        name: skillName,
        data: { version_id: versionId },
        params,
      }),
    decideProposal: (versionId: string, action: DecisionRequestAction) =>
      decide.mutate({
        name: skillName,
        versionId,
        data: {
          action,
          edited_body: action === "apply_edited" ? decisionDraft : null,
        },
        params,
      }),
    reportOutcome: (versionId: string, result: OutcomeReportRequestOutcome) =>
      outcome.mutate({
        name: skillName,
        data: { version_id: versionId, outcome: result, detail: "" },
        params,
      }),
    saveEdit: (body: string, description: string, keepAutoImprove: boolean) =>
      edit.mutate({
        name: skillName,
        data: {
          body,
          description,
          triggers: selectedVersion?.triggers ?? [],
          keep_auto_improve: keepAutoImprove,
          allowed_pattern_classes: [],
          expected_version_id: editBaseVersionId,
        },
        params,
      }),
    excludeSource: (sourceKind: string, sourceRef: string) =>
      exclude.mutate({ sourceKind, sourceRef }),
  };
}
