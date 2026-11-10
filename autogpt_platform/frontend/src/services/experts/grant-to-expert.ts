import { grantExpertCredentials } from "@/app/api/__generated__/endpoints/experts/experts";
import type { QueryClient } from "@tanstack/react-query";
import { invalidateExpertGrantQueries } from "./invalidate-experts";

/** Grants one credential to an expert and refreshes every list that shows
 *  it. Returns false when the grant did not land; the caller keeps the
 *  credential and tells the user, since the connection itself succeeded. */
export async function grantToExpert(
  queryClient: QueryClient,
  expertId: string,
  credentialId: string,
): Promise<boolean> {
  try {
    const response = await grantExpertCredentials(expertId, {
      credential_ids: [credentialId],
    });
    if (response.status !== 200) return false;
  } catch {
    return false;
  }
  await invalidateExpertGrantQueries(queryClient, expertId);
  return true;
}
