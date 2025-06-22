// auth.spec.ts
import { test } from "./fixtures";
import { loginUser, logoutUser } from "./utils/auth";

test.describe("Authentication", () => {
  test("user can login successfully", async ({ page, testUser }) => {
    await loginUser(page, testUser.email, testUser.password);
  });

  test("user can logout successfully", async ({ page, testUser }) => {
    await loginUser(page, testUser.email, testUser.password);
    await logoutUser(page);
  });

  test("login in, then out, then in again", async ({ page, testUser }) => {
    await loginUser(page, testUser.email, testUser.password);
    await logoutUser(page);
    await loginUser(page, testUser.email, testUser.password);
  });
});
