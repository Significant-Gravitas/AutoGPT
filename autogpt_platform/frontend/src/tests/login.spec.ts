import { test, expect } from "@playwright/test";

// test("get started link", async ({ page }) => {
//   await page.goto("https://playwright.dev/");

//   // Click the get started link.
//   await page.getByRole("link", { name: "Get started" }).click();

//   // Expects page to have a heading with the name of Installation.
//   await expect(
//     page.getByRole("heading", { name: "Installation" }),
//   ).toBeVisible();
// });

// import { test, expect } from "@playwright/test";

test("test", async ({ page }) => {
  await page.goto("http://localhost:3000/");
  await page.getByRole("link", { name: "Log In" }).click();
  await page.getByPlaceholder("user@email.com").click();
  await page.getByPlaceholder("user@email.com").fill("test7@ntindle.com");
  await page.getByPlaceholder("user@email.com").press("Tab");
  await page.getByPlaceholder("password").fill("459034902904923904293");
  await page.getByLabel("I agree to the Terms of").click();
  await page.getByRole("button", { name: "Sign up" }).click();

  await expect(page.getByText("test7@ntindle.com")).toBeVisible();
});
