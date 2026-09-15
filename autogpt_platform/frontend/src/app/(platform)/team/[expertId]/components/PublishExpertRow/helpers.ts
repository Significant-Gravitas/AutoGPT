import type { Expert } from "@/app/api/__generated__/models/expert";
import type { ExpertPackagePreview } from "@/app/api/__generated__/models/expertPackagePreview";

/** The publish route builds the package itself, so the dialog is a
 *  confirmation rather than an editor: this is the expert as it stands,
 *  shaped like the preview an upload would have produced. */
export function buildPreviewFromExpert(expert: Expert): ExpertPackagePreview {
  return {
    manifest: {
      identity: {
        name: expert.name,
        role: expert.role,
        tagline: expert.tagline ?? undefined,
      },
    },
    avatar_kind: expert.avatar_url ? "url" : "none",
    skills: expert.skills.map((name) => ({
      slug: name,
      name,
      description: "",
      files: [],
    })),
    workflows: expert.workflows.map((workflow, index) => ({
      index,
      name: workflow.name ?? "Untitled workflow",
      // Only an agent that already has a marketplace listing can travel in a
      // template, which is exactly what the publish route checks for.
      source: workflow.store_listing_version_id ? "store" : "graph",
      store_listing_version_id: workflow.store_listing_version_id,
      schedule_cron: workflow.schedule_cron,
    })),
  };
}

interface UnpublishedWorkflowsDetail {
  code?: string;
  workflows?: string[];
}

/** The publish route answers a blocked publish with the agents to fix, so the
 *  toast can name them instead of repeating a generic failure. */
export function getPublishErrorMessage(error: unknown, name: string): string {
  if (!(error instanceof Error)) return `Couldn't publish ${name}`;

  const response = (error as { response?: { detail?: unknown } }).response;
  const detail = response?.detail as UnpublishedWorkflowsDetail | undefined;

  if (detail?.code === "unpublished_workflows") {
    const workflows = (detail.workflows ?? []).join(", ");
    return `Publish these agents to the marketplace first: ${workflows}`;
  }

  if ((error as { status?: number }).status === 403) {
    return "Only admins can publish";
  }

  return error.message || `Couldn't publish ${name}`;
}
