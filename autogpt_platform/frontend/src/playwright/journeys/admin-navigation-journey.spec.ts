import {
  adminGrantUnavailable,
  grantAdminRole,
  revokeAdminRole,
} from "./admin-role";
import { expect, test } from "./fixtures";
import { authHeaders } from "./journey-user";

// The admin sidebar's links (AdminSidebar/helpers.ts). Every admin page load
// prefetches each of them, and the middleware checks the admin's role for each.
const ADMIN_LINKS = [
  "/admin/users",
  "/admin/marketplace",
  "/admin/spending",
  "/admin/diagnostics",
  "/admin/platform-costs",
  "/admin/execution-analytics",
  "/admin/bots",
  "/admin/block-cost-estimates",
  "/admin/memory",
  "/admin/settings",
];

// A dozen admin page loads' worth of prefetches: more role checks in a row
// than /api/auth/get-session answers for one address (100, see
// docs/platform/rate-limits.md). On a stack served over http, every check
// used to ask it, and the admin was sent to Home (#17308).
const PAGE_LOADS = 12;

test.describe("admin navigation journey", () => {
  const noAdmin = adminGrantUnavailable();
  test.skip(Boolean(noAdmin), noAdmin);

  test("an admin moving quickly between admin pages is never sent away", async ({
    page,
    journeyUser,
  }) => {
    try {
      await grantAdminRole(page, journeyUser);
      await page.goto("/admin/settings");
      await expect(
        page.getByRole("heading", { name: "Admin Settings" }),
      ).toBeVisible();

      // As the router prefetches them, but from an address of this test's
      // own: a check that did look the session up would spend this test's
      // rate limit, not the one the other journeys on the stack share.
      const headers = {
        ...authHeaders(),
        RSC: "1",
        "Next-Router-Prefetch": "1",
      };
      const sentAway: string[] = [];
      const otherAnswers: string[] = [];
      for (let load = 0; load < PAGE_LOADS; load++) {
        const responses = await Promise.all(
          ADMIN_LINKS.map((link) =>
            page.request.get(link, { headers, maxRedirects: 0 }),
          ),
        );
        responses.forEach((response, index) => {
          const location = response.headers()["location"];
          const status = response.status();
          if (location) sentAway.push(`${ADMIN_LINKS[index]} -> ${location}`);
          else if (status === 401 || status === 403)
            sentAway.push(`${ADMIN_LINKS[index]} answered ${status}`);
          // A slow page isn't a failed role check; name it if the test fails.
          else if (!response.ok())
            otherAnswers.push(`${ADMIN_LINKS[index]} answered ${status}`);
        });
      }
      expect(
        sentAway,
        `the admin should pass every role check${
          otherAnswers.length
            ? ` (other answers: ${otherAnswers.join(", ")})`
            : ""
        }`,
      ).toEqual([]);

      await page.reload();
      await expect(
        page.getByRole("heading", { name: "Admin Settings" }),
        "the admin should still reach the page they are on",
      ).toBeVisible();
    } finally {
      await revokeAdminRole(journeyUser);
    }
  });
});
