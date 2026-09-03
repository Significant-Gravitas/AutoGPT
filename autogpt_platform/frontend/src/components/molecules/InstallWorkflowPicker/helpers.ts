export type InstallWorkflowSource = "library" | "marketplace";

export const INSTALL_WORKFLOW_SOURCES: {
  id: InstallWorkflowSource;
  label: string;
}[] = [
  { id: "library", label: "Your agents" },
  { id: "marketplace", label: "Marketplace" },
];

export type WorkflowInstallData =
  | { library_agent_id: string }
  | { store_listing_version_id: string };
