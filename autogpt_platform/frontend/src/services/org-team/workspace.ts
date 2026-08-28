export function getWorkspaceDownloadHref(
  fileId: string,
  organizationId: string | null,
  teamId: string | null,
) {
  const params = new URLSearchParams();
  if (organizationId) params.set("organizationId", organizationId);
  params.set("teamId", teamId ?? "");
  return `/api/workspace/files/${encodeURIComponent(fileId)}/download?${params.toString()}`;
}
