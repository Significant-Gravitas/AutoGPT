import type { Request, Route } from "@playwright/test";
import { expect, test } from "./coverage-fixture";
import { advanceToRoleStep } from "./utils/onboarding";
import { signupTestUser } from "./utils/signup";

const DRAFT_PATH = "/api/proxy/api/onboarding";

// Other PATCHes to the same path (the wallet's, say) carry no draft.
function isDraftSave(request: Request) {
  return (
    request.method() === "PATCH" &&
    new URL(request.url()).pathname === DRAFT_PATH &&
    Boolean(request.postDataJSON()?.wizardProgress)
  );
}

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
      if (lost || !isDraftSave(route.request())) return route.fallback();
      lost = true;
      await route.fetch();
      await route.abort("failed");
    },
  );
  // The answers to the draft saves sent after the lost one, in order.
  const answers: number[] = [];
  page.on("response", (response) => {
    if (lost && isDraftSave(response.request()))
      answers.push(response.status());
  });

  await advanceToRoleStep(page);

  await expect.poll(() => answers, { timeout: 30000 }).toContain(200);

  // Kept by the server, not only in this device's local copy of the draft.
  await expect
    .poll(async () => {
      const saved = await page.request.get(
        new URL(DRAFT_PATH, page.url()).href,
      );
      return (await saved.json()).wizardProgress?.currentStep;
    })
    .toBe("role");
  // The server holds the last step, and the wizard sends one save at a time,
  // so every draft save before that one has been answered, a 409 included.
  expect(answers).not.toContain(409);
  await expect(page.getByText(/updated in another session/i)).toHaveCount(0);
});
