import { ApiError } from "@/lib/api/client";
import {
  grantExpertCredentials,
  integrationKeys,
  revokeExpertCredential,
  type ExpertCredential,
} from "@/lib/api/integrations";
import { queryClient } from "@/lib/session";

function settle(expertID: string, granted: ExpertCredential[]) {
  queryClient.setQueryData(integrationKeys.expertCredentials(expertID), granted);
  // A grant can be what a scheduled workflow was waiting for.
  void queryClient.invalidateQueries({ queryKey: integrationKeys.expertSetup });
}

/** Lets the expert use these accounts. Granting one twice is harmless. */
export async function allowForExpert(expertID: string, credentialIDs: string[]) {
  const granted = await grantExpertCredentials(expertID, credentialIDs);
  settle(expertID, granted);
  return granted;
}

/** Takes an account away from the expert; the account itself stays. */
export async function removeFromExpert(expertID: string, credentialID: string) {
  const granted = await revokeExpertCredential(expertID, credentialID);
  settle(expertID, granted);
  return granted;
}

export function allowErrorText(error: unknown) {
  return error instanceof ApiError && error.status === 402
    ? "Allowing access needs an active plan. Finish this on the web."
    : "Couldn't allow access. Try again.";
}
