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
 */
export function isUserCreator(
  creator: string,
  currentUser: { email?: string } | null,
): boolean {
  if (!currentUser?.email) return false;

  const userHandle = currentUser.email.split("@")[0]?.toLowerCase() || "";
  return creator.toLowerCase().includes(userHandle);
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
