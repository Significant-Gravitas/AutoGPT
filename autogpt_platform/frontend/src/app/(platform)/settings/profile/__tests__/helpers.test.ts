import { describe, expect, test } from "vitest";

import {
  getInitials,
  isFormDirty,
  makeLinkRow,
  MAX_BIO_LENGTH,
  MAX_LINKS,
  profileToFormState,
  validateForm,
  type ProfileFormState,
} from "../helpers";

function rows(...values: string[]) {
  return values.map((v) => makeLinkRow(v));
}

function makeState(
  overrides: Partial<ProfileFormState> = {},
): ProfileFormState {
  return {
    name: "Jane Doe",
    username: "jane_doe",
    description: "I build agents",
    avatar_url: "https://cdn.example.com/avatar.png",
    links: rows("https://jane.dev", "", ""),
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

    expect(state.username).toBe("jane");
    expect(state.name).toBe("Jane");
    expect(state.description).toBe("bio");
    expect(state.avatar_url).toBe("https://cdn.example.com/a.png");
    expect(state.links.map((l) => l.value)).toEqual([
      "https://jane.dev",
      "",
      "",
    ]);
    expect(state.links.every((l) => typeof l.id === "string")).toBe(true);
    const ids = state.links.map((l) => l.id);
    expect(new Set(ids).size).toBe(ids.length);
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
    expect(state.links.map((l) => l.value)).toEqual(["", "", ""]);
  });

  test("preserves links above the initial slot count up to MAX_LINKS", () => {
    const state = profileToFormState({
      username: "j",
      name: "J",
      description: "",
      avatar_url: "",
      links: ["a", "b", "c", "d"],
      is_featured: false,
    });

    expect(state.links.map((l) => l.value)).toEqual(["a", "b", "c", "d"]);
  });

  test("truncates incoming links to MAX_LINKS", () => {
    const tooMany = Array.from(
      { length: MAX_LINKS + 3 },
      (_, i) => `https://x${i}.dev`,
    );
    const state = profileToFormState({
      username: "j",
      name: "J",
      description: "",
      avatar_url: "",
      links: tooMany,
      is_featured: false,
    });

    expect(state.links).toHaveLength(MAX_LINKS);
    expect(state.links.map((l) => l.value)).toEqual(
      tooMany.slice(0, MAX_LINKS),
    );
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
      links: rows("https://jane.dev", "", "", ""),
    });
    expect(isFormDirty(initial, next)).toBe(false);
  });

  test("detects a new non-empty link", () => {
    const next = makeState({
      links: rows("https://jane.dev", "https://github.com/jane", ""),
    });
    expect(isFormDirty(initial, next)).toBe(true);
  });

  test("detects a removed non-empty link", () => {
    const next = makeState({ links: rows("", "", "") });
    expect(isFormDirty(initial, next)).toBe(true);
  });

  test("detects a reordered link", () => {
    const base = makeState({
      links: rows("https://a.dev", "https://b.dev", ""),
    });
    const next = makeState({
      links: rows("https://b.dev", "https://a.dev", ""),
    });
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

  test("preserves emoji and non-BMP characters as a single code point", () => {
    // Surrogate pair: U+1F98A (🦊) is two UTF-16 units.
    expect(getInitials("🦊enny")).toBe("🦊E");
    expect(getInitials("🦊enny Smith")).toBe("🦊S");
  });
});
