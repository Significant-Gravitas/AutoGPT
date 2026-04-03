import { describe, expect, it, vi } from "vitest";
import {
  countSupportedTypes,
  getSupportedTypes,
  getCredentialTypeLabel,
  getActionButtonText,
  getCredentialDisplayName,
  isSystemCredential,
  filterSystemCredentials,
  getSystemCredentials,
  processCredentialDeletion,
  findExistingHostCredentials,
  hasExistingHostCredential,
} from "../helpers";

describe("countSupportedTypes", () => {
  it("returns 0 when nothing is supported", () => {
    expect(countSupportedTypes(false, false, false, false)).toBe(0);
  });

  it("returns 1 for a single supported type", () => {
    expect(countSupportedTypes(true, false, false, false)).toBe(1);
    expect(countSupportedTypes(false, true, false, false)).toBe(1);
  });

  it("returns count of all true flags", () => {
    expect(countSupportedTypes(true, true, true, true)).toBe(4);
    expect(countSupportedTypes(true, false, true, false)).toBe(2);
  });
});

describe("getSupportedTypes", () => {
  it("returns empty array when nothing supported", () => {
    expect(getSupportedTypes(false, false, false, false)).toEqual([]);
  });

  it("returns oauth2 when supportsOAuth2 is true", () => {
    expect(getSupportedTypes(true, false, false, false)).toEqual(["oauth2"]);
  });

  it("returns all supported types in order", () => {
    expect(getSupportedTypes(true, true, true, true)).toEqual([
      "oauth2",
      "api_key",
      "user_password",
      "host_scoped",
    ]);
  });

  it("returns only the enabled types", () => {
    expect(getSupportedTypes(false, true, false, true)).toEqual([
      "api_key",
      "host_scoped",
    ]);
  });
});

describe("getCredentialTypeLabel", () => {
  it("returns 'OAuth' for oauth2", () => {
    expect(getCredentialTypeLabel("oauth2")).toBe("OAuth");
  });

  it("returns 'API Key' for api_key", () => {
    expect(getCredentialTypeLabel("api_key")).toBe("API Key");
  });

  it("returns 'Password' for user_password", () => {
    expect(getCredentialTypeLabel("user_password")).toBe("Password");
  });

  it("returns 'Headers' for host_scoped", () => {
    expect(getCredentialTypeLabel("host_scoped")).toBe("Headers");
  });
});

describe("getActionButtonText", () => {
  it("returns generic text for multiple types without existing", () => {
    expect(getActionButtonText(true, true, false, false, false)).toBe(
      "Add credential",
    );
  });

  it("returns generic text for multiple types with existing", () => {
    expect(getActionButtonText(true, true, false, false, true)).toBe(
      "Add another credential",
    );
  });

  it("returns specific text for single OAuth2 without existing", () => {
    expect(getActionButtonText(true, false, false, false, false)).toBe(
      "Add account",
    );
  });

  it("returns specific text for single OAuth2 with existing", () => {
    expect(getActionButtonText(true, false, false, false, true)).toBe(
      "Connect another account",
    );
  });

  it("returns API key text for single API key", () => {
    expect(getActionButtonText(false, true, false, false, false)).toBe(
      "Add API key",
    );
    expect(getActionButtonText(false, true, false, false, true)).toBe(
      "Use a new API key",
    );
  });

  it("returns password text for single user_password", () => {
    expect(getActionButtonText(false, false, true, false, false)).toBe(
      "Add username and password",
    );
    expect(getActionButtonText(false, false, true, false, true)).toBe(
      "Add a new username and password",
    );
  });

  it("returns headers text for single host_scoped", () => {
    expect(getActionButtonText(false, false, false, true, false)).toBe(
      "Add headers",
    );
    expect(getActionButtonText(false, false, false, true, true)).toBe(
      "Update headers",
    );
  });

  it("returns fallback text when no type is supported", () => {
    expect(getActionButtonText(false, false, false, false, false)).toBe(
      "Add credentials",
    );
    expect(getActionButtonText(false, false, false, false, true)).toBe(
      "Add new credentials",
    );
  });
});

describe("getCredentialDisplayName", () => {
  it("returns title when present", () => {
    expect(getCredentialDisplayName({ title: "My API Key" }, "Google")).toBe(
      "My API Key",
    );
  });

  it("returns username when title is missing", () => {
    expect(
      getCredentialDisplayName({ username: "user@example.com" }, "Google"),
    ).toBe("user@example.com");
  });

  it("returns fallback when both are missing", () => {
    expect(getCredentialDisplayName({}, "Google")).toBe("Your Google account");
  });
});

describe("isSystemCredential", () => {
  it("returns true when is_system is true", () => {
    expect(isSystemCredential({ is_system: true })).toBe(true);
  });

  it("returns false when is_system is false and no title", () => {
    expect(isSystemCredential({ is_system: false })).toBe(false);
  });

  it("returns true when title contains 'system'", () => {
    expect(isSystemCredential({ title: "System Default" })).toBe(true);
  });

  it("returns true when title starts with 'use credits for'", () => {
    expect(isSystemCredential({ title: "Use Credits for OpenAI" })).toBe(true);
  });

  it("returns true when title contains 'use credits'", () => {
    expect(isSystemCredential({ title: "Please use credits" })).toBe(true);
  });

  it("returns false for regular credential", () => {
    expect(isSystemCredential({ title: "My API Key" })).toBe(false);
  });

  it("returns false when title is null", () => {
    expect(isSystemCredential({ title: null })).toBe(false);
  });
});

describe("filterSystemCredentials", () => {
  it("removes system credentials", () => {
    const creds = [
      { title: "My Key", is_system: false },
      { title: "System Default", is_system: true },
      { title: "Other Key" },
    ];
    expect(filterSystemCredentials(creds)).toEqual([
      { title: "My Key", is_system: false },
      { title: "Other Key" },
    ]);
  });

  it("returns empty array when all are system", () => {
    expect(filterSystemCredentials([{ is_system: true }])).toEqual([]);
  });
});

describe("getSystemCredentials", () => {
  it("returns only system credentials", () => {
    const creds = [
      { title: "My Key", is_system: false },
      { title: "System Default", is_system: true },
    ];
    expect(getSystemCredentials(creds)).toEqual([
      { title: "System Default", is_system: true },
    ]);
  });
});

describe("processCredentialDeletion", () => {
  const cred = { id: "cred-1", title: "My Key" };

  it("clears state on successful deletion", async () => {
    const deleteFn = vi.fn().mockResolvedValue({ deleted: true });
    const state = await processCredentialDeletion(
      cred,
      "other",
      deleteFn,
      false,
    );
    expect(state.credentialToDelete).toBeNull();
    expect(state.shouldUnselectCurrent).toBe(false);
  });

  it("flags shouldUnselectCurrent when selected credential is deleted", async () => {
    const deleteFn = vi.fn().mockResolvedValue({ deleted: true });
    const state = await processCredentialDeletion(
      cred,
      "cred-1",
      deleteFn,
      false,
    );
    expect(state.shouldUnselectCurrent).toBe(true);
  });

  it("returns warning when confirmation needed", async () => {
    const deleteFn = vi.fn().mockResolvedValue({
      deleted: false,
      need_confirmation: true,
      message: "In use",
    });
    const state = await processCredentialDeletion(
      cred,
      undefined,
      deleteFn,
      false,
    );
    expect(state.warningMessage).toBe("In use");
    expect(state.credentialToDelete).toBe(cred);
  });

  it("uses fallback warning when message is empty", async () => {
    const deleteFn = vi.fn().mockResolvedValue({
      deleted: false,
      need_confirmation: true,
      message: "",
    });
    const state = await processCredentialDeletion(
      cred,
      undefined,
      deleteFn,
      false,
    );
    expect(state.warningMessage).toBe(
      "This credential is in use. Force delete?",
    );
  });

  it("passes force=true to the delete function", async () => {
    const deleteFn = vi.fn().mockResolvedValue({ deleted: true });
    await processCredentialDeletion(cred, undefined, deleteFn, true);
    expect(deleteFn).toHaveBeenCalledWith("cred-1", true);
  });
});

describe("findExistingHostCredentials", () => {
  const creds = [
    { id: "1", type: "host_scoped", host: "a.com" },
    { id: "2", type: "api_key" },
    { id: "3", type: "host_scoped", host: "b.com" },
  ];

  it("returns matching host_scoped credentials", () => {
    expect(findExistingHostCredentials(creds, "a.com")).toEqual([
      { id: "1", type: "host_scoped", host: "a.com" },
    ]);
  });

  it("returns empty when no match", () => {
    expect(findExistingHostCredentials(creds, "c.com")).toEqual([]);
  });
});

describe("hasExistingHostCredential", () => {
  const creds = [{ type: "host_scoped", host: "x.com" }, { type: "api_key" }];

  it("returns true for existing host", () => {
    expect(hasExistingHostCredential(creds, "x.com")).toBe(true);
  });

  it("returns false for non-existing host", () => {
    expect(hasExistingHostCredential(creds, "y.com")).toBe(false);
  });
});
