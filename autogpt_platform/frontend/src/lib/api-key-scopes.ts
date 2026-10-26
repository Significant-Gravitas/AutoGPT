import type { APIKeyPermission } from "@/app/api/__generated__/models/aPIKeyPermission";

export type ScopeAPIVersion = "v2" | "both" | "v1";

export const SCOPE_DESCRIPTIONS: Record<APIKeyPermission, string> = {
  IDENTITY: "View your user ID, e-mail, and timezone",
  EXECUTE_GRAPH: "Run your workflows",
  READ_GRAPH: "View your workflows and their configurations",
  WRITE_GRAPH: "Create and edit your workflows, and choose which version runs",
  WRITE_LIBRARY:
    "Add, organize, fork and remove workflows in your library, including their schedules and triggers",
  EXECUTE_BLOCK: "Execute individual blocks",
  READ_BLOCK: "View available blocks",
  READ_STORE: "Access the Marketplace",
  WRITE_STORE: "Create, update, and delete marketplace submissions",
  READ_LIBRARY: "View your library workflows and folders",
  RUN_AGENT: "Run workflows from your library",
  READ_RUN: "View run details and history",
  WRITE_RUN: "Stop and delete runs",
  SHARE_RUN: "Share and unshare workflow runs",
  READ_RUN_REVIEW: "View pending human-in-the-loop reviews",
  WRITE_RUN_REVIEW: "Submit human-in-the-loop review responses",
  READ_SCHEDULE: "View your schedules",
  WRITE_SCHEDULE: "Create and delete schedules",
  READ_CREDITS:
    "View your credit balance, transactions, invoices and subscription",
  READ_FILES: "List and download workspace files",
  WRITE_FILES: "Upload and delete workspace files",
  USE_TOOLS:
    "Search and fetch web pages for you; web searches count against your usage allowance",
  MANAGE_INTEGRATIONS: "Set up new integrations",
  READ_INTEGRATIONS: "View your connected integrations",
  DELETE_INTEGRATIONS: "Remove connected integrations",
};

// Which version of the public API checks each scope, read from its routes.
export const SCOPE_API_VERSION: Record<APIKeyPermission, ScopeAPIVersion> = {
  IDENTITY: "both",
  EXECUTE_GRAPH: "v1",
  READ_GRAPH: "both",
  WRITE_GRAPH: "both",
  EXECUTE_BLOCK: "v1",
  READ_BLOCK: "both",
  WRITE_LIBRARY: "both",
  USE_TOOLS: "both",
  MANAGE_INTEGRATIONS: "both",
  READ_INTEGRATIONS: "both",
  DELETE_INTEGRATIONS: "both",
  READ_SCHEDULE: "v2",
  WRITE_SCHEDULE: "v2",
  READ_STORE: "v2",
  WRITE_STORE: "v2",
  READ_LIBRARY: "v2",
  RUN_AGENT: "v2",
  READ_RUN: "v2",
  WRITE_RUN: "v2",
  SHARE_RUN: "v2",
  READ_RUN_REVIEW: "v2",
  WRITE_RUN_REVIEW: "v2",
  READ_CREDITS: "v2",
  READ_FILES: "v2",
  WRITE_FILES: "v2",
};

export const SCOPE_GROUPS: { version: ScopeAPIVersion; label: string }[] = [
  { version: "v2", label: "API v2" },
  { version: "both", label: "API v1 and v2" },
  { version: "v1", label: "API v1 only (stops working 2027-02-01)" },
];
