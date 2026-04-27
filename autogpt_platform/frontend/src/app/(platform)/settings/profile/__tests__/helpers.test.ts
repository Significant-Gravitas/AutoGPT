import { describe, expect, test } from "vitest";

import {
  getInitials,
  isFormDirty,
  MAX_BIO_LENGTH,
  profileToFormState,
  validateForm,
  type ProfileFormState,
} from "../helpers";

function makeState(
  overrides: Partial<ProfileFormState> = {},
): ProfileFormState {
  return {
    name: "Jane Doe",
    username: "jane_doe",
    description: "I build agents",
    avatar_url: "https://cdn.example.com/avatar.png",
    links: ["https://jane.dev", "", ""],
    ...overrides,
  };
}

describe("helpers / profileToFormState", () => {
  test("maps a fully populated profile and pads links to 3 slots", () => {
    const state = profileToFormState({
      username: "jane",
      name: "Jane",
      description: "bio",
      avatar_url: "https://cdn.example.com/a.png",
      links: ["https://jane.dev"],
      is_featured: false,
    });

    expect(state).toEqual({
      username: "jane",
      name: "Jane",
      description: "bio",
      avatar_url: "https://cdn.example.com/a.png",
      links: ["https://jane.dev", "", ""],
    });
  });

  test("falls back to empty strings when fields are nullish", () => {
    const state = profileToFormState({
      username: null as unknown as string,
      name: null as unknown as string,
      description: null as unknown as string,
      avatar_url: null,
      links: null as unknown as string[],
      is_featured: false,
    });

    expect(state.name).toBe("");
    expect(state.username).toBe("");
    expect(state.description).toBe("");
    expect(state.avatar_url).toBe("");
    expect(state.links).toEqual(["", "", ""]);
  });

  test("preserves links when already at or above the initial slot count", () => {
    const state = profileToFormState({
      username: "j",
      name: "J",
      description: "",
      avatar_url: "",
      links: ["a", "b", "c", "d"],
      is_featured: false,
    });

    expect(state.links).toEqual(["a", "b", "c", "d"]);
  });
});

describe("helpers / validateForm", () => {
  test("accepts a valid state", () => {
    const result = validateForm(makeState());
    expect(result.valid).toBe(true);
    expect(result.errors).toEqual({});
  });

  test("flags missing display name", () => {
    const result = validateForm(makeState({ name: "   " }));
    expect(result.valid).toBe(false);
    expect(result.errors.name).toMatch(/required/i);
  });

  test("flags display name longer than 50 chars", () => {
    const result = validateForm(makeState({ name: "x".repeat(51) }));
    expect(result.errors.name).toMatch(/under 50/i);
  });

  test("flags missing handle", () => {
    const result = validateForm(makeState({ username: "" }));
    expect(result.errors.username).toMatch(/required/i);
  });

  test("flags handle that does not match the regex", () => {
    const result = validateForm(makeState({ username: "no spaces here" }));
    expect(result.errors.username).toMatch(/2.+30/);
  });

  test("flags bio longer than the max length", () => {
    const result = validateForm(
      makeState({ description: "x".repeat(MAX_BIO_LENGTH + 1) }),
    );
    expect(result.errors.description).toContain(String(MAX_BIO_LENGTH));
  });

  test("allows bio at exactly the max length", () => {
    const result = validateForm(
      makeState({ description: "x".repeat(MAX_BIO_LENGTH) }),
    );
    expect(result.errors.description).toBeUndefined();
  });
});

describe("helpers / isFormDirty", () => {
  const initial = makeState();

  test("returns false for an unchanged state", () => {
    expect(isFormDirty(initial, makeState())).toBe(false);
  });

  test("detects a changed name", () => {
    expect(isFormDirty(initial, makeState({ name: "Jane Smith" }))).toBe(true);
  });

  test("detects a changed username", () => {
    expect(isFormDirty(initial, makeState({ username: "other" }))).toBe(true);
  });

  test("detects a changed bio", () => {
    expect(isFormDirty(initial, makeState({ description: "different" }))).toBe(
      true,
    );
  });

  test("detects a changed avatar url", () => {
    expect(isFormDirty(initial, makeState({ avatar_url: "x" }))).toBe(true);
  });

  test("ignores empty link slots when comparing", () => {
    const next = makeState({
      links: ["https://jane.dev", "", "", ""],
    });
    expect(isFormDirty(initial, next)).toBe(false);
  });

  test("detects a new non-empty link", () => {
    const next = makeState({
      links: ["https://jane.dev", "https://github.com/jane", ""],
    });
    expect(isFormDirty(initial, next)).toBe(true);
  });

  test("detects a removed non-empty link", () => {
    const next = makeState({ links: ["", "", ""] });
    expect(isFormDirty(initial, next)).toBe(true);
  });

  test("detects a reordered link", () => {
    const base = makeState({ links: ["https://a.dev", "https://b.dev", ""] });
    const next = makeState({ links: ["https://b.dev", "https://a.dev", ""] });
    expect(isFormDirty(base, next)).toBe(true);
  });
});

describe("helpers / getInitials", () => {
  test("returns ? for an empty string", () => {
    expect(getInitials("")).toBe("?");
    expect(getInitials("   ")).toBe("?");
  });

  test("returns the first two letters of a single name, uppercased", () => {
    expect(getInitials("jane")).toBe("JA");
    expect(getInitials("X")).toBe("X");
  });

  test("returns first + last initial for multi-word names", () => {
    expect(getInitials("Jane Doe")).toBe("JD");
    expect(getInitials("ada b. lovelace")).toBe("AL");
  });
});
