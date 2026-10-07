interface ChangeEmailContext {
  path?: string;
  context?: {
    session?: { user?: { id?: string } } | null;
    adapter?: {
      deleteMany: (args: {
        model: string;
        where: Array<{
          field: string;
          value: string;
          operator?: "eq" | "starts_with";
        }>;
      }) => Promise<unknown>;
    };
  };
}

/**
 * A reset or set-password link is a token naming the account, not the address
 * it was mailed to, and opening one verifies the account's current address
 * (onPasswordReset). An unverified account changes its address at once, so
 * without this a link mailed to its old address would verify the new one, and
 * a verified address lets Better Auth link that address's OAuth sign-in into
 * the account. Its outstanding links are dropped when the address changes, so
 * a link only ever verifies the address it reached.
 */
export async function revokeResetLinksOnEmailChange(
  data: { email?: string },
  ctx: ChangeEmailContext | null,
) {
  if (!data.email || ctx?.path !== "/change-email") return;
  const userId = ctx.context?.session?.user?.id;
  const adapter = ctx.context?.adapter;
  if (!userId || !adapter) return;
  await adapter.deleteMany({
    model: "verification",
    where: [
      {
        field: "identifier",
        operator: "starts_with",
        value: "reset-password:",
      },
      { field: "value", value: userId },
    ],
  });
}
