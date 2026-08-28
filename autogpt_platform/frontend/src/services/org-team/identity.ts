export function getTenantEntityKey(
  entityId: string,
  organizationId?: string | null,
  teamId?: string | null,
): string {
  return JSON.stringify([entityId, organizationId ?? null, teamId ?? null]);
}
