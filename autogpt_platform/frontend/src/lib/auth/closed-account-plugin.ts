import type { BetterAuthPlugin } from "better-auth";
import { APIError } from "better-auth/api";
import {
  ACCOUNT_DELETED_BAN_REASON,
  CLOSED_ACCOUNT_ERROR_CODE,
  CLOSED_ACCOUNT_MESSAGE,
} from "./closed-account";

// The admin plugin's fields on a Better Auth user.
interface UserWithBan {
  id: string;
  banned?: boolean | null;
  banReason?: string | null;
}

// The admin plugin answers every ban with one message, so a deleted account
// is refused here first, as closed. Listed before admin() so its hook runs
// first; any other ban falls through to the admin plugin.
export function closedAccount() {
  return {
    id: "closed-account",
    init() {
      return {
        options: {
          databaseHooks: {
            session: {
              create: {
                async before(session, ctx) {
                  if (!ctx) return;
                  const user = await ctx.context.internalAdapter.findUserById(
                    session.userId,
                  );
                  if (isClosedAccount(user)) {
                    throw new APIError("FORBIDDEN", {
                      message: CLOSED_ACCOUNT_MESSAGE,
                      code: CLOSED_ACCOUNT_ERROR_CODE,
                    });
                  }
                },
              },
            },
          },
        },
      };
    },
  } satisfies BetterAuthPlugin;
}

function isClosedAccount(user: UserWithBan | null) {
  return user?.banned === true && user.banReason === ACCOUNT_DELETED_BAN_REASON;
}
