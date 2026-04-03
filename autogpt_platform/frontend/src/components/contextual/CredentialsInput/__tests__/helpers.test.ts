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
  resolveActionTarget,
  headerPairsToRecord,
  addHeaderPairToList,
  removeHeaderPairFromList,
  updateHeaderPairInList,
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

describe("resolveActionTarget", () => {
  it("returns type_selector when hasMultipleCredentialTypes is true", () => {
    expect(resolveActionTarget(true, true, true, false, false)).toBe(
      "type_selector",
    );
  });

  it("returns oauth when only OAuth2 is supported", () => {
    expect(resolveActionTarget(false, true, false, false, false)).toBe("oauth");
  });

  it("returns api_key when only API key is supported", () => {
    expect(resolveActionTarget(false, false, true, false, false)).toBe(
      "api_key",
    );
  });

  it("returns user_password when only user_password is supported", () => {
    expect(resolveActionTarget(false, false, false, true, false)).toBe(
      "user_password",
    );
  });

  it("returns host_scoped when only host_scoped is supported", () => {
    expect(resolveActionTarget(false, false, false, false, true)).toBe(
      "host_scoped",
    );
  });

  it("returns null when nothing is supported", () => {
    expect(resolveActionTarget(false, false, false, false, false)).toBeNull();
  });

  it("prefers oauth over api_key when not multiple types", () => {
    expect(resolveActionTarget(false, true, true, false, false)).toBe("oauth");
  });
});

describe("headerPairsToRecord", () => {
  it("converts pairs to record filtering empty entries", () => {
    const pairs = [
      { key: "Authorization", value: "Bearer token" },
      { key: "", value: "ignored" },
      { key: "X-Key", value: "" },
      { key: "  Accept  ", value: "  application/json  " },
    ];
    expect(headerPairsToRecord(pairs)).toEqual({
      Authorization: "Bearer token",
      Accept: "application/json",
    });
  });

  it("returns empty object for empty pairs", () => {
    expect(headerPairsToRecord([])).toEqual({});
  });

  it("returns empty object when all pairs are empty", () => {
    expect(headerPairsToRecord([{ key: "", value: "" }])).toEqual({});
  });
});

describe("addHeaderPairToList", () => {
  it("adds a new empty pair to the list", () => {
    const pairs = [{ key: "a", value: "b" }];
    const result = addHeaderPairToList(pairs);
    expect(result).toHaveLength(2);
    expect(result[1]).toEqual({ key: "", value: "" });
  });

  it("does not mutate the original array", () => {
    const pairs = [{ key: "a", value: "b" }];
    const result = addHeaderPairToList(pairs);
    expect(pairs).toHaveLength(1);
    expect(result).not.toBe(pairs);
  });
});

describe("removeHeaderPairFromList", () => {
  it("removes the pair at the given index", () => {
    const pairs = [
      { key: "a", value: "1" },
      { key: "b", value: "2" },
      { key: "c", value: "3" },
    ];
    const result = removeHeaderPairFromList(pairs, 1);
    expect(result).toEqual([
      { key: "a", value: "1" },
      { key: "c", value: "3" },
    ]);
  });

  it("does not remove when only one pair remains", () => {
    const pairs = [{ key: "a", value: "1" }];
    const result = removeHeaderPairFromList(pairs, 0);
    expect(result).toHaveLength(1);
    expect(result).toBe(pairs);
  });

  it("does not mutate the original array", () => {
    const pairs = [
      { key: "a", value: "1" },
      { key: "b", value: "2" },
    ];
    removeHeaderPairFromList(pairs, 0);
    expect(pairs).toHaveLength(2);
  });
});

describe("updateHeaderPairInList", () => {
  it("updates the key of a pair at the given index", () => {
    const pairs = [
      { key: "a", value: "1" },
      { key: "b", value: "2" },
    ];
    const result = updateHeaderPairInList(pairs, 0, "key", "updated");
    expect(result[0]).toEqual({ key: "updated", value: "1" });
    expect(result[1]).toEqual({ key: "b", value: "2" });
  });

  it("updates the value of a pair at the given index", () => {
    const pairs = [{ key: "a", value: "1" }];
    const result = updateHeaderPairInList(pairs, 0, "value", "new-val");
    expect(result[0]).toEqual({ key: "a", value: "new-val" });
  });

  it("does not mutate the original array or pair objects", () => {
    const pairs = [{ key: "a", value: "1" }];
    const result = updateHeaderPairInList(pairs, 0, "key", "b");
    expect(pairs[0].key).toBe("a");
    expect(result).not.toBe(pairs);
    expect(result[0]).not.toBe(pairs[0]);
  });
});
