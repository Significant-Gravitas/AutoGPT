import {
  getTeamScopedQueryKey,
  getTenantRequestInit,
} from "@/components/contextual/TeamPicker/helpers";
import { ApiError } from "@/lib/autogpt-server-api/helpers";

export const ORG_CREDITS_MANAGED_MESSAGE =
  "Your organization's credits are managed by its owner and billing managers.";

// Matches the backend's 409 for a team organization's top-up
// (TEAM_ORG_TOP_UP_REFUSED in billing/credits/routes.py).
export const TEAM_ORG_TOP_UP_REFUSED_MESSAGE =
  "A team organization's credits are added by AutoGPT for now, so they can't be bought here.";

export interface OrgWalletScope {
  organizationID: string | null;
  isTeamOrg: boolean;
}

// The credits routes answer 403 to members without MANAGE_BILLING, which in a
// team organization means everyone but its owner and billing managers.
export function isForbiddenError(error: unknown): boolean {
  return error instanceof ApiError && error.status === 403;
}

function retryUnlessForbidden(failureCount: number, error: unknown) {
  return !isForbiddenError(error) && failureCount < 3;
}

// Pins a credits request to the active team organization's wallet. A
// personal organization gets no override, so its requests stay as they were.
export function getOrgWalletRequest({
  organizationID,
  isTeamOrg,
}: OrgWalletScope) {
  return isTeamOrg ? getTenantRequestInit(organizationID, null) : undefined;
}

// The request plus a cache key of the team organization's own, so its wallet
// never shares a cache entry with a personal one.
export function getOrgWalletQueryOptions(
  queryKey: readonly unknown[],
  scope: OrgWalletScope,
) {
  if (!scope.isTeamOrg) return { request: undefined, query: {} };
  return {
    request: getOrgWalletRequest(scope),
    query: {
      queryKey: getTeamScopedQueryKey(queryKey, scope.organizationID, null),
      retry: retryUnlessForbidden,
    },
  };
}

// A person's plan, payment method and invoices belong to their own account,
// which the backend reaches through their personal organization. In a team
// organization, send no org so these requests resolve there.
export function getPersonalBillingQueryOptions(
  queryKey: readonly unknown[],
  { isTeamOrg }: OrgWalletScope,
) {
  if (!isTeamOrg) return { request: undefined, query: {} };
  return {
    request: getTenantRequestInit(null, null),
    query: { queryKey: getTeamScopedQueryKey(queryKey, null, null) },
  };
}
