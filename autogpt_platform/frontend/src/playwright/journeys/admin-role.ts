import { expect, type Page } from "@playwright/test";
import { withJourneyDatabase } from "./journey-database";
import { isLocalStack, type JourneyUser, signIn } from "./journey-user";

/**
 * Why this run can't make an admin, or undefined when it can. Making one
 * writes the stack's database (see withJourneyDatabase), which a journey
 * reaches on a local stack, and on a deployed one only through
 * PLAYWRIGHT_JOURNEY_DATABASE_URL.
 */
export function adminGrantUnavailable(): string | undefined {
  if (isLocalStack() || process.env.PLAYWRIGHT_JOURNEY_DATABASE_URL) {
    return undefined;
  }
  return (
    "making an admin writes the stack's database: set " +
    "PLAYWRIGHT_JOURNEY_DATABASE_URL to the deployed stack's"
  );
}

/**
 * Makes a journey user a platform admin, the way an operator does: the role
 * on their auth user (`UserAuthIdentity`), which Better Auth's admin plugin
 * reads and the JWT the backend checks carries. A session caches its user for
 * five minutes, so the user signs in again for one that has the role.
 */
export async function grantAdminRole(page: Page, user: JourneyUser) {
  const userID = user.id;
  if (!userID) throw new Error(`no user id for ${user.email} to make an admin`);
  await withJourneyDatabase("the admin grant", async (client) => {
    const result = await client.query(
      `UPDATE platform."UserAuthIdentity" SET role = 'admin' WHERE id = $1`,
      [userID],
    );
    if (result.rowCount !== 1) {
      throw new Error(`no auth user ${userID} to make an admin`);
    }
  });
  await signIn(page, user);
  const response = await page.request.get("/api/auth/get-session");
  expect(
    response.ok(),
    `reading the admin's session: ${response.status()}`,
  ).toBe(true);
  const session = (await response.json()) as {
    user?: { role?: string | null };
  } | null;
  expect(
    session?.user?.role,
    "the admin's new session should carry the role",
  ).toBe("admin");
}
