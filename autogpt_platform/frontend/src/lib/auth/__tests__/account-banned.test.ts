import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { signOutIfRefused } from "../account-banned";

describe("signOutIfRefused", () => {
  const replace = vi.fn();

  beforeEach(() => {
    vi.stubGlobal("location", { pathname: "/settings/api-keys", replace });
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    replace.mockReset();
  });

  it("signs a banned account out with the banned notice", () => {
    signOutIfRefused(403, { detail: { code: "account_banned" } });

    expect(replace).toHaveBeenCalledWith("/logout?reason=banned");
  });

  it("signs out a session that has ended, without the banned notice", () => {
    signOutIfRefused(401, { detail: { code: "session_revoked" } });

    expect(replace).toHaveBeenCalledWith("/logout");
  });

  it("leaves every other error to the caller", () => {
    signOutIfRefused(401, { detail: "Not authenticated" });
    signOutIfRefused(403, { detail: { code: "session_revoked" } });
    signOutIfRefused(500, null);

    expect(replace).not.toHaveBeenCalled();
  });

  it("does nothing once already on /logout", () => {
    vi.stubGlobal("location", { pathname: "/logout", replace });

    signOutIfRefused(401, { detail: { code: "session_revoked" } });

    expect(replace).not.toHaveBeenCalled();
  });
});
