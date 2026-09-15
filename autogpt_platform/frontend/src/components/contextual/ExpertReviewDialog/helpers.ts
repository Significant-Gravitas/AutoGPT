import type { ExpertPackagePreview } from "@/app/api/__generated__/models/expertPackagePreview";
import type { WorkflowResolution } from "@/app/api/__generated__/models/workflowResolution";

/** The `edits` form field of `POST /api/experts/import`. FastAPI reads it as a
 *  JSON string, so the backend's `ExpertImportEdits` is not in the schema and
 *  has no generated twin — keep these names in step with `package_import.py`. */
export interface ExpertImportEdits {
  name?: string;
  removed_skill_slugs: string[];
  removed_workflow_indices: number[];
  workflows: { index: number; schedule_enabled: boolean }[];
}

export interface ExpertReviewDraft {
  name: string;
  removedSkillSlugs: string[];
  removedWorkflowIndices: number[];
  /** Workflows whose schedule should start. Only ones that ship a cron. */
  scheduledIndices: number[];
}

export function buildDraftFromPreview(
  preview: ExpertPackagePreview | null,
): ExpertReviewDraft {
  return {
    name: preview?.manifest.identity.name ?? "",
    removedSkillSlugs: [],
    removedWorkflowIndices: [],
    // A packaged schedule starts on: the file said this expert runs that way,
    // and a silently paused workflow is the harder surprise of the two.
    scheduledIndices: (preview?.workflows ?? [])
      .filter((workflow) => Boolean(workflow.schedule_cron))
      .map((workflow) => workflow.index),
  };
}

export function buildEditsFromDraft(
  draft: ExpertReviewDraft,
  preview: ExpertPackagePreview | null,
): ExpertImportEdits {
  const kept = (preview?.workflows ?? []).filter(
    (workflow) =>
      workflow.schedule_cron &&
      !draft.removedWorkflowIndices.includes(workflow.index),
  );

  return {
    name: draft.name.trim(),
    removed_skill_slugs: [...draft.removedSkillSlugs],
    removed_workflow_indices: [...draft.removedWorkflowIndices],
    workflows: kept.map((workflow) => ({
      index: workflow.index,
      schedule_enabled: draft.scheduledIndices.includes(workflow.index),
    })),
  };
}

/** Add or drop one member of a "what the user left out" list. */
export function toggleMember<T>(list: T[], value: T): T[] {
  return list.includes(value)
    ? list.filter((item) => item !== value)
    : [...list, value];
}

export function serializeExpertEdits(edits: ExpertImportEdits): string {
  return JSON.stringify(edits);
}

/** Why the primary button is disabled, phrased for the user, or null when it
 *  is not. Publishing needs every kept agent on the marketplace already —
 *  a template nobody else can install is worse than no template. */
export function getBlockingReason(
  mode: "import" | "publish",
  draft: ExpertReviewDraft,
  preview: ExpertPackagePreview | null,
): string | null {
  if (!draft.name.trim()) return "Give this expert a name.";

  const errors = preview?.errors ?? [];
  if (errors.length > 0) return "Fix the problems above to continue.";

  if (mode === "publish") {
    const unpublished = (preview?.workflows ?? []).filter(
      (workflow) =>
        workflow.source !== "store" &&
        !draft.removedWorkflowIndices.includes(workflow.index),
    );
    if (unpublished.length > 0) {
      const names = unpublished.map((workflow) => workflow.name).join(", ");
      return `Publish these agents to the marketplace first: ${names}`;
    }
  }

  return null;
}

export function getWorkflowSourceLabel(source: WorkflowResolution["source"]) {
  if (source === "store")
    return { label: "Marketplace", variant: "info" } as const;
  if (source === "graph")
    return { label: "From file", variant: "info" } as const;
  return { label: "Can't import", variant: "error" } as const;
}
