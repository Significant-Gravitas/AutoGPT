import { redirect } from "next/navigation";
import { beforeEach, describe, expect, it, vi } from "vitest";
import RateLimitsPage from "../page";

vi.mock("next/navigation", () => ({ redirect: vi.fn() }));

beforeEach(() => vi.mocked(redirect).mockReset());

describe("/admin/rate-limits", () => {
  it("sends a bookmark of the old page to the users page", async () => {
    await RateLimitsPage({ searchParams: Promise.resolve({}) });

    expect(redirect).toHaveBeenCalledWith("/admin/users");
  });

  it.each([
    [
      { user_id: "11111111-1111-4111-8111-111111111111" },
      "11111111-1111-4111-8111-111111111111",
    ],
    [{ email: "first_last@example.com" }, "first_last%40example.com"],
    [{ user: "first_last@example.com" }, "first_last%40example.com"],
    [{ user_id: ["first-id", "second-id"] }, "first-id"],
  ])("carries the user in %o", async (searchParams, expected) => {
    await RateLimitsPage({ searchParams: Promise.resolve(searchParams) });

    expect(redirect).toHaveBeenCalledWith(`/admin/users?user=${expected}`);
  });
});
