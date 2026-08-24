import { environment } from "@/services/environment";
import { z } from "zod";

import { mintServiceToken } from "@/lib/auth/service-token";
import type {
  ProvisionedPartnerIdentity,
  VerifiedPartnerIdentity,
} from "./types";

const provisionResponseSchema = z.object({
  user_id: z.string().uuid(),
  organization_id: z.string().uuid(),
  team_id: z.string().uuid(),
});

export async function provisionPartnerIdentity(
  identity: VerifiedPartnerIdentity,
): Promise<ProvisionedPartnerIdentity> {
  const serviceToken = await mintServiceToken("partner-embed:provision");
  const response = await fetch(
    `${environment.getAGPTServerBaseUrl()}/api/embed/v1/provision`,
    {
      method: "POST",
      headers: {
        Authorization: `Bearer ${serviceToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        partner_id: identity.partnerID,
        external_subject: identity.externalSubject,
        external_account_id: identity.externalAccountID,
        email: identity.email,
        display_name: identity.displayName,
        account_name: identity.accountName,
        is_admin: identity.isAdmin,
      }),
    },
  );
  if (!response.ok) {
    throw new Error(`Partner provisioning failed with ${response.status}`);
  }
  const result = provisionResponseSchema.parse(await response.json());
  return {
    userID: result.user_id,
    organizationID: result.organization_id,
    teamID: result.team_id,
  };
}
