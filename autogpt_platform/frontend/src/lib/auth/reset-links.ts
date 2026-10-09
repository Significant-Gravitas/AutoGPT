import type { AuthEmailContext } from "./auth-email-cooldown";
import {
  resetLinkAddresses,
  resetLinkAddressIdentifier,
} from "./reset-link-address";

const RESET_LINK = "reset-password:";

/**
 * A reset or set-password link is a token naming the account, not the address
 * it was mailed to. An unverified account changes its address at once, so a
 * link mailed to its old address could verify the new one, and a verified
 * address lets Better Auth link that address's OAuth sign-in into the account.
 * And after any change of address, the old address should no longer reset the
 * account's password.
 *
 * So after every update to the account, whatever path made it (/change-email,
 * a confirmed change on /verify-email, an admin update), its links mailed to
 * another address are dropped. A link with no address recorded (recording is
 * best-effort) is kept: the reset checks the address itself
 * (reset-link-address.ts), so such a link resets the password but verifies
 * nothing.
 */
export async function revokeResetLinksMailedElsewhere(
  context: AuthEmailContext,
  user: { id: string; email: string },
) {
  const links = (await context.adapter.findMany({
    model: "verification",
    where: [
      { field: "identifier", value: RESET_LINK, operator: "starts_with" },
      { field: "value", value: user.id },
    ],
    limit: 100,
  })) as Array<{ identifier: string }>;
  const mailedTo = await resetLinkAddresses(context, user.id);
  const email = user.email.toLowerCase();
  for (const { identifier } of links) {
    const address = mailedTo.get(identifier.slice(RESET_LINK.length));
    if (address === undefined || address === email) continue;
    await deleteVerification(context, identifier);
  }
  for (const [token, address] of mailedTo) {
    if (address === email) continue;
    await deleteVerification(
      context,
      resetLinkAddressIdentifier(user.id, token),
    );
  }
}

function deleteVerification(context: AuthEmailContext, identifier: string) {
  return context.adapter.deleteMany({
    model: "verification",
    where: [{ field: "identifier", value: identifier }],
  });
}
