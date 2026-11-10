import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { handleRefusedAccount, signOutIfRefused } from "../account-banned";

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

describe("handleRefusedAccount", () => {
  const replace = vi.fn();

  beforeEach(() => {
    vi.stubGlobal("location", { pathname: "/settings/api-keys", replace });
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    replace.mockReset();
  });

  it.each([
    [
      "a banned account",
      403,
      { detail: { code: "account_banned" } },
      "/logout?reason=banned",
    ],
    [
      "an ended session",
      401,
      { detail: { code: "session_revoked" } },
      "/logout",
    ],
    [
      "a closed account",
      401,
      { detail: "This account is closed." },
      "/account-closed",
    ],
  ])("sends %s to %s", (_, status, body, path) => {
    handleRefusedAccount(status, body);

    expect(replace).toHaveBeenCalledTimes(1);
    expect(replace).toHaveBeenCalledWith(path);
  });

  it("leaves every other error to the caller", () => {
    handleRefusedAccount(401, { detail: "Not authenticated" });
    handleRefusedAccount(403, { detail: "This account is closed." });
    handleRefusedAccount(500, null);

    expect(replace).not.toHaveBeenCalled();
  });
});
