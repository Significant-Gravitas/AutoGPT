// TEMPORARY proof for #15904, on a proof branch only. Never merge.
import { expect, test } from "./fixtures";
import { JOURNEY_USER_AGENT } from "./user-agent";

const POPUP = { name: "What's new" } as const;

test.describe("changelog proof", () => {
  test("with the shared fixture, a warm changelog opens no popup", async ({
    page,
    journeyUser,
  }) => {
    expect(journeyUser.id).toBeTruthy();
    const warm = await page.request.get("/api/changelog");
    expect(warm.ok(), "the changelog should answer").toBe(true);
    expect(await warm.text(), "the changelog should list entries").toContain(
      "/docs/platform/changelog/changelog/",
    );

    await page.goto("/home");
    await expect(
      page.getByRole("textbox", { name: "Chat message input" }),
    ).toBeVisible();
    for (let i = 0; i < 24; i++) {
      expect(
        await page.getByRole("dialog", POPUP).count(),
        `no popup ${i * 250} ms after the composer showed`,
      ).toBe(0);
      await page.waitForTimeout(250);
    }
  });

  test("without the seen marker, the same page opens the popup", async ({
    page,
    journeyUser,
    browser,
  }, testInfo) => {
    expect(journeyUser.id).toBeTruthy();
    await page.request.get("/api/changelog");
    const { cookies } = await page.context().storageState();
    const control = await browser.newContext({
      baseURL: testInfo.project.use.baseURL,
      userAgent: JOURNEY_USER_AGENT,
      storageState: { cookies, origins: [] },
    });
    const controlPage = await control.newPage();

    await controlPage.goto("/home");
    await expect(controlPage.getByRole("dialog", POPUP)).toBeVisible();
    await control.close();
  });
});
