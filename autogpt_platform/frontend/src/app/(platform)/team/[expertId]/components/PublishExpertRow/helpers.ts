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
      // A missing stored listing does not mean the agent is unpublished: the
      // admin may have published it after attaching it, which never writes
      // back to the workflow. The publish route looks that listing up itself
      // and names the agents it cannot find, so the dialog only confirms.
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
 *  off the wire, and `ApiError.response` is deliberately unshaped.
 *
 *  `null` covers "nothing nameable here", which includes a body that carries
 *  the code but no usable list — naming no agents at all reads worse than the
 *  generic failure it would otherwise fall through to. */
function readUnpublishedWorkflows(response: unknown): string[] | null {
  if (typeof response !== "object" || response === null) return null;
  const detail = (response as { detail?: unknown }).detail;
  if (typeof detail !== "object" || detail === null) return null;

  const { code, workflows } = detail as { code?: unknown; workflows?: unknown };
  if (code !== "unpublished_workflows") return null;
  if (!Array.isArray(workflows)) return null;

  const named = workflows.filter(
    (workflow): workflow is string => typeof workflow === "string",
  );
  return named.length > 0 ? named : null;
}

/** A refusal the route is supposed to produce: the expert has a private agent,
 *  or the caller is not an admin. The toast explains both, so neither is worth
 *  a Sentry event. */
export function isExpectedPublishRefusal(error: unknown): boolean {
  if (!(error instanceof ApiError)) return false;
  if (error.status === 403) return true;
  return readUnpublishedWorkflows(error.response) !== null;
}
