import {
  type AuthEmailContext,
  createVerification,
} from "./auth-email-cooldown";

// `reset-password-mailed-to:<userId>:<token>` holds the address the link went
// to. The user id comes first so an account's records can be read by prefix
// (reset-links.ts); the token is used up before the reset's hook runs, so the
// record is found by its suffix.
const MAILED_TO = "reset-password-mailed-to:";

interface ResetUser {
  id: string;
  email: string;
  emailVerified?: boolean | null;
}

export interface ResetLinkContext extends AuthEmailContext {
  internalAdapter: {
    findUserById: (userId: string) => Promise<ResetUser | null>;
    updateUser: (
      userId: string,
      data: { emailVerified: boolean },
    ) => Promise<unknown>;
  };
}

/**
 * Better Auth binds a reset token to the account (`reset-password:<token>`
 * holds the user id), not to the address the link was mailed to, and an
 * unverified account can change its address on the spot. So each link
 * records where it was mailed, and a reset verifies the account's address
 * only while the two still match. Links are also dropped once the address
 * changes (reset-links.ts), but one issued while a change is landing can
 * outlive that.
 */
export function recordResetLinkAddress(
  context: AuthEmailContext,
  link: {
    token: string;
    userId: string;
    email: string;
    expiresInSeconds: number;
  },
) {
  return createVerification(context, {
    identifier: resetLinkAddressIdentifier(link.userId, link.token),
    value: link.email.toLowerCase(),
    expiresInSeconds: link.expiresInSeconds,
  });
}

// Runs after /reset-password has used up the token and saved the password.
export async function verifyAddressTheResetLinkWasMailedTo(
  context: ResetLinkContext,
  token: string,
) {
  // The adapter's LIKE leaves `_` (in base64url tokens) a wildcard, so the
  // suffix is matched exactly here.
  const records = (await context.adapter.findMany({
    model: "verification",
    where: [
      { field: "identifier", value: `:${token}`, operator: "ends_with" },
      { field: "identifier", value: MAILED_TO, operator: "starts_with" },
    ],
    limit: 10,
  })) as Array<{ identifier: string; value: string }>;
  const record = records.find(
    ({ identifier }) =>
      identifier.startsWith(MAILED_TO) && identifier.endsWith(`:${token}`),
  );
  if (!record) return;
  await context.adapter.deleteMany({
    model: "verification",
    where: [{ field: "identifier", value: record.identifier }],
  });
  const userId = record.identifier.slice(MAILED_TO.length, -(token.length + 1));
  const user = await context.internalAdapter.findUserById(userId);
  if (!user || user.emailVerified) return;
  if (user.email.toLowerCase() !== record.value) return;
  await context.internalAdapter.updateUser(user.id, { emailVerified: true });
}

export function resetLinkAddressIdentifier(userId: string, token: string) {
  return `${MAILED_TO}${userId}:${token}`;
}

// The addresses the account's outstanding links were mailed to, by token.
export async function resetLinkAddresses(
  context: AuthEmailContext,
  userId: string,
) {
  const prefix = `${MAILED_TO}${userId}:`;
  const records = (await context.adapter.findMany({
    model: "verification",
    where: [{ field: "identifier", value: prefix, operator: "starts_with" }],
    limit: 100,
  })) as Array<{ identifier: string; value: string }>;
  return new Map(
    records
      .filter(({ identifier }) => identifier.startsWith(prefix))
      .map((record) => [record.identifier.slice(prefix.length), record.value]),
  );
}
