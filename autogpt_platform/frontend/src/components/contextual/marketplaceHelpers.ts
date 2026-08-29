/**
 * Marketplace-specific helper functions that can be reused across different marketplace screens
 */

/**
 * Calculate the latest marketplace version from agent graph versions
 */
export function getLatestMarketplaceVersion(
  agentGraphVersions?: string[],
): number | undefined {
  if (!agentGraphVersions?.length) return undefined;

  return Math.max(...agentGraphVersions.map((v: string) => parseInt(v, 10)));
}

/**
 * Check if the current user is the creator of the agent
 * Uses ID-based comparison for accurate matching
 */
export function isUserCreator(
  creatorId: string | undefined,
  currentUserId: string | undefined,
): boolean {
  if (!creatorId || !currentUserId) return false;
  return creatorId === currentUserId;
}

/**
 * Calculate update status for an agent
 */
export function calculateUpdateStatus({
  latestMarketplaceVersion,
  currentVersion,
  isUserCreator,
  isAgentAddedToLibrary,
}: {
  latestMarketplaceVersion?: number;
  currentVersion: number;
  isUserCreator: boolean;
  isAgentAddedToLibrary: boolean;
}) {
  if (!latestMarketplaceVersion) {
    return { hasUpdate: false, hasUnpublishedChanges: false };
  }

  const hasUnpublishedChanges =
    isUserCreator &&
    isAgentAddedToLibrary &&
    currentVersion > latestMarketplaceVersion;

  const hasUpdate =
    isAgentAddedToLibrary &&
    !isUserCreator &&
    latestMarketplaceVersion > currentVersion;

  return { hasUpdate, hasUnpublishedChanges };
}
