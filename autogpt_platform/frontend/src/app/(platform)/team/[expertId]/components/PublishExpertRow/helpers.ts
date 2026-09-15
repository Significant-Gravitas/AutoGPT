import type { Expert } from "@/app/api/__generated__/models/expert";
import type { ExpertPackagePreview } from "@/app/api/__generated__/models/expertPackagePreview";
import { ApiError } from "@/lib/autogpt-server-api/helpers";

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

/** The publish route answers a blocked publish with the agents to fix, so the
 *  toast can name them instead of repeating a generic failure. */
export function getPublishErrorMessage(error: unknown, name: string): string {
  const fallback = `Couldn't publish ${name}`;
  if (!(error instanceof ApiError)) {
    return error instanceof Error && error.message ? error.message : fallback;
  }

  const workflows = readUnpublishedWorkflows(error.response);
  if (workflows) {
    return `Publish these agents to the marketplace first: ${workflows.join(", ")}`;
  }

  if (error.status === 403) return "Only admins can publish";

  return error.message || fallback;
}

/** The 400 body, checked at runtime rather than trusted from a type: it is JSON
 *  off the wire, and `ApiError.response` is deliberately unshaped. */
function readUnpublishedWorkflows(response: unknown): string[] | null {
  if (typeof response !== "object" || response === null) return null;
  const detail = (response as { detail?: unknown }).detail;
  if (typeof detail !== "object" || detail === null) return null;

  const { code, workflows } = detail as { code?: unknown; workflows?: unknown };
  if (code !== "unpublished_workflows") return null;
  return Array.isArray(workflows)
    ? workflows.filter(
        (workflow): workflow is string => typeof workflow === "string",
      )
    : [];
}
