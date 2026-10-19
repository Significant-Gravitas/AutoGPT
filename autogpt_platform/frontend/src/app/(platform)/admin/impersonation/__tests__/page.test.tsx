import { ImpersonationState } from "@/lib/impersonation";
import { redirect } from "next/navigation";
import { beforeEach, describe, expect, it, vi } from "vitest";
import ImpersonationPage from "../page";

vi.mock("next/navigation", () => ({ redirect: vi.fn() }));

const TARGET_ID = "11111111-1111-4111-8111-111111111111";

beforeEach(() => {
  vi.mocked(redirect).mockReset();
  vi.restoreAllMocks();
});

describe("/admin/impersonation", () => {
  it("sends a bookmark of the old page to the users page", async () => {
    vi.spyOn(ImpersonationState, "getServerSide").mockResolvedValue(null);

    await ImpersonationPage({ searchParams: Promise.resolve({}) });

    expect(redirect).toHaveBeenCalledWith("/admin/users");
  });

  it("opens the user being impersonated", async () => {
    vi.spyOn(ImpersonationState, "getServerSide").mockResolvedValue(TARGET_ID);

    await ImpersonationPage({ searchParams: Promise.resolve({}) });

    expect(redirect).toHaveBeenCalledWith(`/admin/users?user=${TARGET_ID}`);
  });

  it("prefers a user named in the link over the one being impersonated", async () => {
    vi.spyOn(ImpersonationState, "getServerSide").mockResolvedValue(TARGET_ID);

    await ImpersonationPage({
      searchParams: Promise.resolve({ user_id: "other-user" }),
    });

    expect(redirect).toHaveBeenCalledWith("/admin/users?user=other-user");
  });
});
