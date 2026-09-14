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
import { SkillVersionSummary } from "@/app/api/__generated__/models/skillVersionSummary";
import { stripFrontmatter } from "@/services/skill-learning/helpers";
import {
  SkillEditDraft,
  SkillReviewState,
  UpdateSkillReview,
} from "./useSkillReviewState";

interface Args {
  expertId: string | null;
  skillName: string;
  state: SkillReviewState;
  update: UpdateSkillReview;
  onChanged: () => void;
}

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
  state,
  update,
  onChanged,
}: Args) {
  const queryClient = useQueryClient();
  const { toast } = useToast();
  const { view, selectedVersionId, editor } = state;

  function updateEditor(patch: Partial<SkillEditDraft>) {
    update((previous) => ({
      editor: previous.editor ? { ...previous.editor, ...patch } : null,
    }));
  }

  const params = scopeParams(expertId);
  const detailParams = selectedVersionId
    ? { ...params, version_id: selectedVersionId }
    : params;
  const detailQuery = useGetSkillLearningDetail(skillName, detailParams, {
    query: { select: (res) => okData(res) ?? null },
  });
  const detail = detailQuery.data ?? null;
  const selectedVersion = selectedVersionId
    ? ((detail?.versions ?? []).find(
        (version) => version.id === selectedVersionId,
      ) ?? null)
    : (detail?.current_version ?? null);

  function refresh() {
    queryClient.invalidateQueries({
      queryKey: getGetSkillLearningDetailQueryKey(skillName),
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
      onSuccess: (res, variables) => {
        const data = okData(res);
        if (data && report(data, "Decision recorded")) {
          update((previous) => ({
            decisions: { ...previous.decisions, [variables.versionId]: "" },
          }));
        }
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
          update({ editor: null, selectedVersionId: null });
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
    setView: (next: SkillReviewState["view"]) => update({ view: next }),
    detail,
    isLoading: detailQuery.isLoading,
    isError:
      detailQuery.isError ||
      Boolean(selectedVersionId && detail && !selectedVersion),
    refetch: detailQuery.refetch,
    selectedVersion,
    selectVersion: (id: string) => update({ selectedVersionId: id }),
    editor,
    updateEditor,
    startEdit: (version: SkillVersionSummary) => {
      update({
        editor: {
          baseVersionId: version.id,
          body: stripFrontmatter(version.body ?? ""),
          description: version.description,
          triggers: version.triggers ?? [],
          keepAutoImprove: false,
        },
      });
    },
    cancelEdit: () => update({ editor: null }),
    reviewCurrent: () => {
      if (detail?.current_version)
        updateEditor({
          baseVersionId: detail.current_version.id,
          triggers: detail.current_version.triggers ?? [],
        });
    },
    decisionDraft: state.decisions[detail?.open_decision?.id ?? ""] ?? "",
    setDecisionDraft: (body: string) => {
      const id = detail?.open_decision?.id;
      if (id)
        update((previous) => ({
          decisions: { ...previous.decisions, [id]: body },
        }));
    },
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
          edited_body:
            action === "apply_edited"
              ? (state.decisions[versionId] ?? null)
              : null,
        },
        params,
      }),
    reportOutcome: (versionId: string, result: OutcomeReportRequestOutcome) =>
      outcome.mutate({
        name: skillName,
        data: { version_id: versionId, outcome: result, detail: "" },
        params,
      }),
    saveEdit: () => {
      if (!editor) return;
      edit.mutate({
        name: skillName,
        data: {
          body: editor.body,
          description: editor.description,
          triggers: editor.triggers,
          keep_auto_improve: editor.keepAutoImprove,
          allowed_pattern_classes: [],
          expected_version_id: editor.baseVersionId,
        },
        params,
      });
    },
    excludeSource: (sourceKind: string, sourceRef: string) =>
      exclude.mutate({ sourceKind, sourceRef }),
  };
}
