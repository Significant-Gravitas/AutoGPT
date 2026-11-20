import type { Route } from "@playwright/test";
import { expect, test } from "./coverage-fixture";
import { advanceToRoleStep } from "./utils/onboarding";
import { signupTestUser } from "./utils/signup";

const DRAFT_PATH = "/api/proxy/api/onboarding";

// #17158: a wizard save can reach the server and be saved there while its
// answer never reaches the page (a mobile connection dropping it, or the
// wizard giving up at 10 s). The wizard's next save was then refused with a
// 409, as if another session had saved.
test("onboarding keeps saving after a save that landed without an answer", async ({
  page,
}) => {
  test.setTimeout(120000);

  await signupTestUser(page, undefined, undefined, false);
  await expect(page).toHaveURL(/\/onboarding/);

  let lost = false;
  await page.route(
    (url) => url.pathname === DRAFT_PATH,
    async (route: Route) => {
      const request = route.request();
      if (request.method() !== "PATCH" || lost) return route.fallback();
      if (!request.postDataJSON()?.wizardProgress) return route.fallback();
      lost = true;
      await route.fetch();
      await route.abort("failed");
    },
  );
  const answered: number[] = [];
  page.on("response", (response) => {
    if (
      response.request().method() === "PATCH" &&
      new URL(response.url()).pathname === DRAFT_PATH
    )
      answered.push(response.status());
  });

  await advanceToRoleStep(page);

  await expect
    .poll(() => answered.includes(200), { timeout: 30000 })
    .toBe(true);
  expect(lost).toBe(true);
  expect(answered).not.toContain(409);
  await expect(page.getByText(/updated in another session/i)).toHaveCount(0);

  // What the page saved after the lost save is the account's own copy.
  await page.reload();
  await expect(page.getByText("What best describes you")).toBeVisible({
    timeout: 30000,
  });
});
