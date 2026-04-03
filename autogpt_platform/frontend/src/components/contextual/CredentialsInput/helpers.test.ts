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
  OAUTH_TIMEOUT_MS,
  MASKED_KEY_LENGTH,
  resolveActionTarget,
  headerPairsToRecord,
  addHeaderPairToList,
  removeHeaderPairFromList,
  updateHeaderPairInList,
} from "./helpers";

describe("countSupportedTypes", () => {
  it("returns 0 when no types are supported", () => {
    expect(countSupportedTypes(false, false, false, false)).toBe(0);
  });

  it("returns 1 when only one type is supported", () => {
    expect(countSupportedTypes(true, false, false, false)).toBe(1);
    expect(countSupportedTypes(false, true, false, false)).toBe(1);
    expect(countSupportedTypes(false, false, true, false)).toBe(1);
    expect(countSupportedTypes(false, false, false, true)).toBe(1);
  });

  it("returns correct count for multiple types", () => {
    expect(countSupportedTypes(true, true, false, false)).toBe(2);
    expect(countSupportedTypes(true, true, true, false)).toBe(3);
    expect(countSupportedTypes(true, true, true, true)).toBe(4);
  });
});

describe("getSupportedTypes", () => {
  it("returns empty array when no types are supported", () => {
    expect(getSupportedTypes(false, false, false, false)).toEqual([]);
  });

  it("returns oauth2 when supportsOAuth2 is true", () => {
    expect(getSupportedTypes(true, false, false, false)).toEqual(["oauth2"]);
  });

  it("returns api_key when supportsApiKey is true", () => {
    expect(getSupportedTypes(false, true, false, false)).toEqual(["api_key"]);
  });

  it("returns user_password when supportsUserPassword is true", () => {
    expect(getSupportedTypes(false, false, true, false)).toEqual([
      "user_password",
    ]);
  });

  it("returns host_scoped when supportsHostScoped is true", () => {
    expect(getSupportedTypes(false, false, false, true)).toEqual([
      "host_scoped",
    ]);
  });

  it("returns all types in order when all are supported", () => {
    expect(getSupportedTypes(true, true, true, true)).toEqual([
      "oauth2",
      "api_key",
      "user_password",
      "host_scoped",
    ]);
  });
});

describe("getCredentialTypeLabel", () => {
  it("returns OAuth for oauth2", () => {
    expect(getCredentialTypeLabel("oauth2")).toBe("OAuth");
  });

  it("returns API Key for api_key", () => {
    expect(getCredentialTypeLabel("api_key")).toBe("API Key");
  });

  it("returns Password for user_password", () => {
    expect(getCredentialTypeLabel("user_password")).toBe("Password");
  });

  it("returns Headers for host_scoped", () => {
    expect(getCredentialTypeLabel("host_scoped")).toBe("Headers");
  });
});

describe("getActionButtonText", () => {
  describe("when multiple types are supported", () => {
    it("returns generic text without existing credentials", () => {
      expect(getActionButtonText(true, true, false, false, false)).toBe(
        "Add credential",
      );
    });

    it("returns generic text with existing credentials", () => {
      expect(getActionButtonText(true, true, false, false, true)).toBe(
        "Add another credential",
      );
    });
  });

  describe("when only OAuth2 is supported", () => {
    it("returns 'Add account' without existing credentials", () => {
      expect(getActionButtonText(true, false, false, false, false)).toBe(
        "Add account",
      );
    });

    it("returns 'Connect another account' with existing credentials", () => {
      expect(getActionButtonText(true, false, false, false, true)).toBe(
        "Connect another account",
      );
    });
  });

  describe("when only API key is supported", () => {
    it("returns 'Add API key' without existing credentials", () => {
      expect(getActionButtonText(false, true, false, false, false)).toBe(
        "Add API key",
      );
    });

    it("returns 'Use a new API key' with existing credentials", () => {
      expect(getActionButtonText(false, true, false, false, true)).toBe(
        "Use a new API key",
      );
    });
  });

  describe("when only user_password is supported", () => {
    it("returns 'Add username and password' without existing credentials", () => {
      expect(getActionButtonText(false, false, true, false, false)).toBe(
        "Add username and password",
      );
    });

    it("returns 'Add a new username and password' with existing credentials", () => {
      expect(getActionButtonText(false, false, true, false, true)).toBe(
        "Add a new username and password",
      );
    });
  });

  describe("when only host_scoped is supported", () => {
    it("returns 'Add headers' without existing credentials", () => {
      expect(getActionButtonText(false, false, false, true, false)).toBe(
        "Add headers",
      );
    });

    it("returns 'Update headers' with existing credentials", () => {
      expect(getActionButtonText(false, false, false, true, true)).toBe(
        "Update headers",
      );
    });
  });

  describe("when no types are supported", () => {
    it("returns 'Add credentials' without existing credentials", () => {
      expect(getActionButtonText(false, false, false, false, false)).toBe(
        "Add credentials",
      );
    });

    it("returns 'Add new credentials' with existing credentials", () => {
      expect(getActionButtonText(false, false, false, false, true)).toBe(
        "Add new credentials",
      );
    });
  });
});

describe("getCredentialDisplayName", () => {
  it("returns title when present", () => {
    expect(
      getCredentialDisplayName({ title: "My Key", username: "user" }, "GitHub"),
    ).toBe("My Key");
  });

  it("falls back to username when title is missing", () => {
    expect(getCredentialDisplayName({ username: "jdoe" }, "GitHub")).toBe(
      "jdoe",
    );
  });

  it("falls back to display name when both title and username are missing", () => {
    expect(getCredentialDisplayName({}, "GitHub")).toBe("Your GitHub account");
  });

  it("falls back when title is empty string", () => {
    expect(getCredentialDisplayName({ title: "" }, "GitHub")).toBe(
      "Your GitHub account",
    );
  });
});

describe("isSystemCredential", () => {
  it("returns true when is_system is true", () => {
    expect(isSystemCredential({ is_system: true })).toBe(true);
  });

  it("returns false when is_system is false and no title", () => {
    expect(isSystemCredential({ is_system: false })).toBe(false);
  });

  it("returns false when title is null", () => {
    expect(isSystemCredential({ title: null })).toBe(false);
  });

  it("returns false when title is absent", () => {
    expect(isSystemCredential({})).toBe(false);
  });

  it("returns true when title contains 'system'", () => {
    expect(isSystemCredential({ title: "System API Key" })).toBe(true);
  });

  it("returns true when title contains 'system' case-insensitively", () => {
    expect(isSystemCredential({ title: "SYSTEM key" })).toBe(true);
  });

  it("returns true when title starts with 'Use credits for'", () => {
    expect(isSystemCredential({ title: "Use credits for OpenAI" })).toBe(true);
  });

  it("returns true when title starts with 'use credits for' case-insensitively", () => {
    expect(isSystemCredential({ title: "use credits for Anthropic" })).toBe(
      true,
    );
  });

  it("returns true when title contains 'use credits'", () => {
    expect(isSystemCredential({ title: "Please use credits here" })).toBe(true);
  });

  it("returns false for a normal credential title", () => {
    expect(isSystemCredential({ title: "My Personal Key" })).toBe(false);
  });
});

describe("filterSystemCredentials", () => {
  it("returns empty array for empty input", () => {
    expect(filterSystemCredentials([])).toEqual([]);
  });

  it("filters out system credentials", () => {
    const credentials = [
      { title: "My Key" },
      { title: "System Key" },
      { title: "Use credits for OpenAI" },
      { title: "Personal Token" },
    ];
    const result = filterSystemCredentials(credentials);
    expect(result).toEqual([{ title: "My Key" }, { title: "Personal Token" }]);
  });

  it("filters out credentials with is_system flag", () => {
    const credentials = [
      { title: "Normal", is_system: false },
      { title: "Hidden", is_system: true },
    ];
    const result = filterSystemCredentials(credentials);
    expect(result).toEqual([{ title: "Normal", is_system: false }]);
  });
});

describe("getSystemCredentials", () => {
  it("returns empty array for empty input", () => {
    expect(getSystemCredentials([])).toEqual([]);
  });

  it("returns only system credentials", () => {
    const credentials = [
      { title: "My Key" },
      { title: "System Key" },
      { title: "Use credits for OpenAI" },
      { title: "Personal Token" },
    ];
    const result = getSystemCredentials(credentials);
    expect(result).toEqual([
      { title: "System Key" },
      { title: "Use credits for OpenAI" },
    ]);
  });

  it("returns credentials with is_system flag", () => {
    const credentials = [
      { title: "Normal", is_system: false },
      { title: "Hidden", is_system: true },
    ];
    const result = getSystemCredentials(credentials);
    expect(result).toEqual([{ title: "Hidden", is_system: true }]);
  });
});

describe("constants", () => {
  it("OAUTH_TIMEOUT_MS is 5 minutes", () => {
    expect(OAUTH_TIMEOUT_MS).toBe(300000);
  });

  it("MASKED_KEY_LENGTH is 15", () => {
    expect(MASKED_KEY_LENGTH).toBe(15);
  });
});

describe("processCredentialDeletion", () => {
  const cred = { id: "cred-1", title: "My Key" };

  it("returns cleared state on successful deletion", async () => {
    const deleteFn = vi.fn().mockResolvedValue({ deleted: true });
    const state = await processCredentialDeletion(
      cred,
      "other-id",
      deleteFn,
      false,
    );

    expect(deleteFn).toHaveBeenCalledWith("cred-1", false);
    expect(state.credentialToDelete).toBeNull();
    expect(state.warningMessage).toBeNull();
    expect(state.shouldUnselectCurrent).toBe(false);
  });

  it("sets shouldUnselectCurrent when deleting the selected credential", async () => {
    const deleteFn = vi.fn().mockResolvedValue({ deleted: true });
    const state = await processCredentialDeletion(
      cred,
      "cred-1",
      deleteFn,
      false,
    );

    expect(state.shouldUnselectCurrent).toBe(true);
    expect(state.credentialToDelete).toBeNull();
  });

  it("returns warning state when confirmation is needed", async () => {
    const deleteFn = vi.fn().mockResolvedValue({
      deleted: false,
      need_confirmation: true,
      message: "Used by 3 agents",
    });
    const state = await processCredentialDeletion(
      cred,
      undefined,
      deleteFn,
      false,
    );

    expect(state.warningMessage).toBe("Used by 3 agents");
    expect(state.credentialToDelete).toBe(cred);
    expect(state.shouldUnselectCurrent).toBe(false);
  });

  it("uses default warning message when none provided", async () => {
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

  it("passes force flag to delete function", async () => {
    const deleteFn = vi.fn().mockResolvedValue({ deleted: true });
    await processCredentialDeletion(cred, undefined, deleteFn, true);

    expect(deleteFn).toHaveBeenCalledWith("cred-1", true);
  });

  it("returns unchanged state for unknown result shape", async () => {
    const deleteFn = vi.fn().mockResolvedValue({ deleted: false });
    const state = await processCredentialDeletion(
      cred,
      undefined,
      deleteFn,
      false,
    );

    expect(state.warningMessage).toBeNull();
    expect(state.credentialToDelete).toBe(cred);
    expect(state.shouldUnselectCurrent).toBe(false);
  });
});

describe("findExistingHostCredentials", () => {
  const credentials = [
    { id: "1", type: "host_scoped", host: "api.example.com" },
    { id: "2", type: "host_scoped", host: "api.other.com" },
    { id: "3", type: "api_key" },
    { id: "4", type: "host_scoped", host: "api.example.com" },
  ];

  it("finds credentials matching the given host", () => {
    const result = findExistingHostCredentials(credentials, "api.example.com");
    expect(result).toHaveLength(2);
    expect(result[0].id).toBe("1");
    expect(result[1].id).toBe("4");
  });

  it("returns empty array when no match", () => {
    expect(findExistingHostCredentials(credentials, "unknown.com")).toEqual([]);
  });

  it("ignores non-host_scoped credentials", () => {
    const result = findExistingHostCredentials(credentials, "api.other.com");
    expect(result).toHaveLength(1);
    expect(result[0].id).toBe("2");
  });

  it("returns empty array for empty credentials list", () => {
    expect(findExistingHostCredentials([], "any.com")).toEqual([]);
  });
});

describe("hasExistingHostCredential", () => {
  const credentials = [
    { type: "host_scoped", host: "api.example.com" },
    { type: "api_key" },
  ];

  it("returns true when a host_scoped credential exists for the host", () => {
    expect(hasExistingHostCredential(credentials, "api.example.com")).toBe(
      true,
    );
  });

  it("returns false when no matching host_scoped credential exists", () => {
    expect(hasExistingHostCredential(credentials, "other.com")).toBe(false);
  });

  it("returns false for empty credentials list", () => {
    expect(hasExistingHostCredential([], "any.com")).toBe(false);
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
});

describe("headerPairsToRecord", () => {
  it("converts non-empty pairs to record", () => {
    const pairs = [
      { key: "Authorization", value: "Bearer token" },
      { key: "", value: "ignored" },
      { key: "X-Key", value: "" },
    ];
    expect(headerPairsToRecord(pairs)).toEqual({
      Authorization: "Bearer token",
    });
  });

  it("trims keys and values", () => {
    expect(
      headerPairsToRecord([{ key: "  Accept  ", value: "  text/html  " }]),
    ).toEqual({ Accept: "text/html" });
  });

  it("returns empty object for empty pairs", () => {
    expect(headerPairsToRecord([])).toEqual({});
  });
});

describe("addHeaderPairToList", () => {
  it("appends an empty pair", () => {
    const result = addHeaderPairToList([{ key: "a", value: "b" }]);
    expect(result).toHaveLength(2);
    expect(result[1]).toEqual({ key: "", value: "" });
  });
});

describe("removeHeaderPairFromList", () => {
  it("removes the pair at index", () => {
    const pairs = [
      { key: "a", value: "1" },
      { key: "b", value: "2" },
    ];
    expect(removeHeaderPairFromList(pairs, 0)).toEqual([
      { key: "b", value: "2" },
    ]);
  });

  it("does not remove the last pair", () => {
    const pairs = [{ key: "a", value: "1" }];
    expect(removeHeaderPairFromList(pairs, 0)).toBe(pairs);
  });
});

describe("updateHeaderPairInList", () => {
  it("updates key at the given index", () => {
    const pairs = [{ key: "a", value: "1" }];
    const result = updateHeaderPairInList(pairs, 0, "key", "b");
    expect(result[0]).toEqual({ key: "b", value: "1" });
  });

  it("updates value at the given index", () => {
    const pairs = [{ key: "a", value: "1" }];
    const result = updateHeaderPairInList(pairs, 0, "value", "2");
    expect(result[0]).toEqual({ key: "a", value: "2" });
  });

  it("does not mutate originals", () => {
    const pairs = [{ key: "a", value: "1" }];
    updateHeaderPairInList(pairs, 0, "key", "b");
    expect(pairs[0].key).toBe("a");
  });
});
