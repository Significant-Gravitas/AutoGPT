const PERSONAL_ORGANIZATION = "__personal__";
const ORGANIZATION_HOME = "__org_home__";

interface BuilderHrefOptions {
  graphId?: string | null;
  graphVersion?: number | string | null;
  executionId?: string | null;
  view?: string | null;
  organizationId: string | null;
  teamId: string | null;
}

export interface BuilderTenantScope {
  organizationId: string | null;
  teamId: string | null;
}

export function getBuilderHref({
  graphId,
  graphVersion,
  executionId,
  view,
  organizationId,
  teamId,
}: BuilderHrefOptions): string {
  const params = new URLSearchParams();
  if (graphId) params.set("flowID", graphId);
  if (graphVersion !== null && graphVersion !== undefined) {
    params.set("flowVersion", String(graphVersion));
  }
  if (executionId) params.set("flowExecutionID", executionId);
  if (view) params.set("view", view);
  params.set("organizationId", organizationId ?? PERSONAL_ORGANIZATION);
  params.set("teamId", teamId ?? ORGANIZATION_HOME);
  return `/build?${params.toString()}`;
}

export function decodeBuilderTenantScope(
  organizationId: string | null,
  teamId: string | null,
): BuilderTenantScope | null {
  if (organizationId === null && teamId === null) return null;
  return {
    organizationId:
      organizationId === PERSONAL_ORGANIZATION ? null : organizationId,
    teamId: teamId === ORGANIZATION_HOME ? null : teamId,
  };
}

export function getCopilotHref(
  sessionId: string,
  organizationId: string | null,
  teamId: string | null,
): string {
  const params = new URLSearchParams({ sessionId });
  params.set("organizationId", organizationId ?? PERSONAL_ORGANIZATION);
  params.set("teamId", teamId ?? ORGANIZATION_HOME);
  return `/copilot?${params.toString()}`;
}

export function getLibraryAgentHref(
  libraryAgentId: string,
  organizationId: string | null,
  teamId: string | null,
  activeItem?: string | null,
  activeTab?: string | null,
): string {
  const params = new URLSearchParams();
  params.set("organizationId", organizationId ?? PERSONAL_ORGANIZATION);
  params.set("teamId", teamId ?? ORGANIZATION_HOME);
  if (activeTab) params.set("activeTab", activeTab);
  if (activeItem) params.set("activeItem", activeItem);
  return `/library/agents/${encodeURIComponent(libraryAgentId)}?${params.toString()}`;
}

export function getCopilotStartHref(
  organizationId: string | null,
  teamId: string | null,
  prompt?: string,
) {
  const params = new URLSearchParams();
  params.set("organizationId", organizationId ?? PERSONAL_ORGANIZATION);
  params.set("teamId", teamId ?? ORGANIZATION_HOME);
  if (prompt) params.set("autosubmit", "true");
  return `/copilot?${params.toString()}${prompt ? `#prompt=${encodeURIComponent(prompt)}` : ""}`;
}

export function getCopilotExpertHref(
  expertId: string,
  organizationId: string | null,
  teamId: string | null,
  kickoff = false,
) {
  const params = new URLSearchParams({ expertId });
  params.set("organizationId", organizationId ?? PERSONAL_ORGANIZATION);
  params.set("teamId", teamId ?? ORGANIZATION_HOME);
  if (kickoff) params.set("kickoff", "1");
  return `/copilot?${params.toString()}`;
}
