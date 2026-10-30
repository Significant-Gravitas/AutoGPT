import { beforeEach, describe, expect, it, vi } from "vitest";

const actions = vi.hoisted(() => ({
  getCurrentUser: vi.fn(),
  validateSession: vi.fn(),
  refreshSession: vi.fn(),
  serverLogout: vi.fn(),
}));
vi.mock("@/lib/auth/actions", () => actions);
vi.mock("@/lib/react-query/queryClient", () => ({
  getQueryClient: () => ({ clear() {}, invalidateQueries: async () => {} }),
}));
vi.mock("@/services/analytics/reset-identity", () => ({
  resetAnalyticsIdentity() {},
}));

import { useAuthStore } from "@/lib/auth/hooks/useAuthStore";
import type { User } from "@/lib/auth/types";
import { getSystemHeaders, ImpersonationState } from "@/lib/impersonation";
import { beginImpersonation } from "../beginImpersonation";

const admin = { id: "admin-1", email: "a@x.invalid", role: "admin" } as User;

function auditThatWaits() {
  let finish!: () => void;
  const audit = new Promise<void>((resolve) => {
    finish = resolve;
  });
  return { send: () => audit, finish };
}

beforeEach(() => {
  sessionStorage.clear();
  localStorage.clear();
  ImpersonationState.clear();
  useAuthStore.setState({ user: admin, hasLoadedUser: true, apiRef: null });
  actions.serverLogout.mockResolvedValue({ success: true });
});

describe("beginImpersonation", () => {
  it("records the target for the admin once the audit alert lands", async () => {
    expect(await beginImpersonation("target-a", async () => undefined)).toBe(
      "started",
    );
    expect(getSystemHeaders()).toEqual({ "X-Act-As-User-Id": "target-a" });
  });

  it("records nothing if the admin signed out while the alert was in flight", async () => {
    const audit = auditThatWaits();
    const starting = beginImpersonation("target-a", audit.send);

    await useAuthStore.getState().logOut();
    audit.finish();

    expect(await starting).toBe("admin-changed");
    expect(ImpersonationState.read()).toBeNull();
    expect(getSystemHeaders()).toEqual({});
  });

  it("records nothing if the admin lost the role while the alert was in flight", async () => {
    const audit = auditThatWaits();
    const starting = beginImpersonation("target-a", audit.send);

    actions.validateSession.mockResolvedValue({
      user: { ...admin, role: "authenticated" },
      isValid: true,
    });
    await useAuthStore.getState().validateSession({
      force: true,
      path: "/admin",
      router: { push() {}, refresh() {} } as never,
    });
    audit.finish();

    expect(await starting).toBe("admin-changed");
    expect(ImpersonationState.read()).toBeNull();
  });

  it("sends no audit alert when no admin is signed in", async () => {
    useAuthStore.setState({ user: null });
    const send = vi.fn(async () => undefined);

    expect(await beginImpersonation("target-a", send)).toBe("no-admin");
    expect(send).not.toHaveBeenCalled();
  });
});
