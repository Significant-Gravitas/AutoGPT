export type InstallWorkflowSource = "library" | "marketplace";

export const INSTALL_WORKFLOW_SOURCES: {
  id: InstallWorkflowSource;
  label: string;
}[] = [
  { id: "library", label: "Your workflows" },
  { id: "marketplace", label: "Marketplace" },
];

export type WorkflowInstallData =
  | { library_agent_id: string }
  | { store_listing_version_id: string };

/** Row subtitle: the workflow's own description, or a neutral source label.
 *  A user's own agent has no marketplace creator, so `creator_name` reads
 *  "Unknown" there. */
export function workflowSubtitle(description: string | null | undefined) {
  return description?.trim() || "From your library";
}
