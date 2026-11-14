import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { handleRefusedAccount, signOutIfRefused } from "../account-banned";

const validateSession = vi.hoisted(() => vi.fn());

vi.mock("../hooks/useAuthStore", () => ({
  useAuthStore: { getState: () => ({ validateSession }) },
}));

const SESSION_REVOKED = {
  detail: { code: "session_revoked", message: "Session expired or revoked" },
};

function settle() {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

describe("signOutIfRefused", () => {
  const replace = vi.fn();

  beforeEach(() => {
    vi.stubGlobal("location", { pathname: "/settings/api-keys", replace });
    validateSession.mockResolvedValue(false);
  });

  afterEach(async () => {
    await settle();
    vi.unstubAllGlobals();
    replace.mockReset();
    validateSession.mockReset();
  });

  it("signs a banned account out with the banned notice", () => {
    signOutIfRefused(403, { detail: { code: "account_banned" } });

    expect(replace).toHaveBeenCalledWith("/logout?reason=banned");
    expect(validateSession).not.toHaveBeenCalled();
  });

  it("checks an ended session with the auth backend instead of signing out itself", async () => {
    signOutIfRefused(401, SESSION_REVOKED);

    await vi.waitFor(() =>
      expect(validateSession).toHaveBeenCalledWith({ force: true }),
    );
    expect(replace).not.toHaveBeenCalled();
  });

  it("starts one check for a burst of refusals", async () => {
    let finish: (valid: boolean) => void = () => undefined;
    validateSession.mockReturnValue(
      new Promise<boolean>((resolve) => {
        finish = resolve;
      }),
    );

    signOutIfRefused(401, SESSION_REVOKED);
    signOutIfRefused(401, SESSION_REVOKED);
    signOutIfRefused(401, SESSION_REVOKED);
    await vi.waitFor(() => expect(validateSession).toHaveBeenCalled());
    signOutIfRefused(401, SESSION_REVOKED);
    await settle();
    expect(validateSession).toHaveBeenCalledTimes(1);

    finish(false);
    await settle();
    signOutIfRefused(401, SESSION_REVOKED);
    await vi.waitFor(() => expect(validateSession).toHaveBeenCalledTimes(2));
  });

  it("keeps a newer live session signed in when a late refusal for an older one arrives", async () => {
    validateSession.mockResolvedValue(true);

    signOutIfRefused(401, SESSION_REVOKED);

    await vi.waitFor(() => expect(validateSession).toHaveBeenCalled());
    await settle();
    expect(replace).not.toHaveBeenCalled();
  });

  it("signs nobody out when the check itself fails", async () => {
    validateSession.mockRejectedValue(new Error("auth backend unreachable"));

    signOutIfRefused(401, SESSION_REVOKED);

    await vi.waitFor(() => expect(validateSession).toHaveBeenCalled());
    await settle();
    expect(replace).not.toHaveBeenCalled();
    signOutIfRefused(401, SESSION_REVOKED);
    await vi.waitFor(() => expect(validateSession).toHaveBeenCalledTimes(2));
  });

  it("leaves every other error to the caller", async () => {
    signOutIfRefused(401, { detail: "Not authenticated" });
    signOutIfRefused(401, { detail: "Session expired or revoked" });
    signOutIfRefused(403, { detail: { code: "session_revoked" } });
    signOutIfRefused(500, null);
    await settle();

    expect(replace).not.toHaveBeenCalled();
    expect(validateSession).not.toHaveBeenCalled();
  });

  it("does nothing more for a banned account once already on /logout", () => {
    vi.stubGlobal("location", { pathname: "/logout", replace });

    signOutIfRefused(403, { detail: { code: "account_banned" } });

    expect(replace).not.toHaveBeenCalled();
  });
});

describe("handleRefusedAccount", () => {
  const replace = vi.fn();

  beforeEach(() => {
    vi.stubGlobal("location", { pathname: "/settings/api-keys", replace });
    validateSession.mockResolvedValue(false);
  });

  afterEach(async () => {
    await settle();
    vi.unstubAllGlobals();
    replace.mockReset();
    validateSession.mockReset();
  });

  it.each([
    [
      "a banned account",
      403,
      { detail: { code: "account_banned" } },
      "/logout?reason=banned",
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
    expect(validateSession).not.toHaveBeenCalled();
  });

  it("checks an ended session rather than navigating", async () => {
    handleRefusedAccount(401, SESSION_REVOKED);

    await vi.waitFor(() =>
      expect(validateSession).toHaveBeenCalledWith({ force: true }),
    );
    expect(replace).not.toHaveBeenCalled();
  });

  it("leaves every other error to the caller", async () => {
    handleRefusedAccount(401, { detail: "Not authenticated" });
    handleRefusedAccount(403, { detail: "This account is closed." });
    handleRefusedAccount(500, null);
    await settle();

    expect(replace).not.toHaveBeenCalled();
    expect(validateSession).not.toHaveBeenCalled();
  });
});
