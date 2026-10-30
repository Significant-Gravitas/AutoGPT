import { expect, test } from "./coverage-fixture";
import { LoginPage } from "./pages/login.page";

test("team happy path: a name and a role create an expert that asks its questions and keeps the answers", async ({
  page,
}) => {
  test.setTimeout(120000);
  const name = `Deps merger ${Date.now()}`;

  await new LoginPage(page).loginAsSeededUser("parallelB");
  await page.goto("/team");
  await page.getByRole("button", { name: "Create an Expert" }).first().click();
  await page.getByLabel("Name").fill(name);
  await page.getByLabel("Role").fill("Release engineer");
  await page.getByRole("button", { name: "Create", exact: true }).click();

  await expect(page).toHaveURL(/\/team\/[0-9a-f-]{36}\?welcome=1$/, {
    timeout: 15000,
  });
  const welcome = page.getByRole("complementary", {
    name: `${name} introduces themself`,
  });
  await expect(welcome.getByText(`Hi, I'm ${name}.`)).toBeVisible();

  const questions = welcome.getByRole("radiogroup");
  await expect(questions.first()).toBeVisible();
  const questionCount = await questions.count();
  expect(questionCount).toBeGreaterThanOrEqual(2);
  expect(questionCount).toBeLessThanOrEqual(3);
  for (let index = 0; index < questionCount; index++) {
    await questions.nth(index).getByRole("radio").first().click();
  }
  await expect(welcome.getByText(/Got it\./)).toBeVisible();

  // The soul is written from the answers in the background; the page picks
  // it up without a reload, replacing the placeholder identity.
  await page.getByRole("tab", { name: /Soul/ }).click();
  await expect(page.getByText("Identity", { exact: true })).toBeVisible();
  await expect(page.getByText(/an AI Expert created by you/)).toHaveCount(0, {
    timeout: 45000,
  });

  // The expert is real: it survives a reload, and the welcome is not asked
  // twice.
  await page.reload();
  await expect(page.getByText(name, { exact: true }).first()).toBeVisible();
  await expect(page.getByRole("button", { name: "Skip for now" })).toHaveCount(
    0,
  );
});
