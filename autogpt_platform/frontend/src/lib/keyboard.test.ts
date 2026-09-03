import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { KEY_NAMES, isComposingEvent, isKey } from "./keyboard";

function keydown(key: string, extra: Partial<KeyboardEvent> = {}) {
  const event = new KeyboardEvent("keydown", { key });
  for (const [name, value] of Object.entries(extra)) {
    Object.defineProperty(event, name, { value });
  }
  return event;
}

function reactKeydown(key: string, extra: Partial<KeyboardEvent> = {}) {
  const nativeEvent = keydown(key, extra);
  return { key, nativeEvent } as unknown as React.KeyboardEvent;
}

describe("isComposingEvent", () => {
  it("is false for a plain key press", () => {
    expect(isComposingEvent(keydown("Enter"))).toBe(false);
  });

  it("is true while the IME reports composition", () => {
    expect(isComposingEvent(keydown("Enter", { isComposing: true }))).toBe(
      true,
    );
  });

  it("is true for the Safari post-composition keyCode 229", () => {
    expect(isComposingEvent(keydown("Enter", { keyCode: 229 }))).toBe(true);
  });

  it("reads the native event behind a React synthetic event", () => {
    expect(isComposingEvent(reactKeydown("Enter", { isComposing: true }))).toBe(
      true,
    );
    expect(isComposingEvent(reactKeydown("Enter"))).toBe(false);
  });
});

describe("isKey", () => {
  it("matches the key when not composing", () => {
    expect(isKey(keydown("Enter"), "Enter")).toBe(true);
    expect(isKey(reactKeydown(" "), " ")).toBe(true);
  });

  it("matches any of several keys", () => {
    expect(isKey(keydown(" "), "Enter", " ")).toBe(true);
    expect(isKey(keydown("Escape"), "Enter", " ")).toBe(false);
  });

  it("does not match a different key", () => {
    expect(isKey(keydown("Escape"), "Enter")).toBe(false);
  });

  it("never matches while composing, even if the key name matches", () => {
    expect(isKey(keydown("Enter", { isComposing: true }), "Enter")).toBe(false);
    expect(isKey(reactKeydown("Enter", { keyCode: 229 }), "Enter")).toBe(false);
  });

  it("matches again once composition is over (Safari Enter flow)", () => {
    expect(isKey(keydown("Enter", { keyCode: 229 }), "Enter")).toBe(false);
    expect(isKey(keydown("Enter", { keyCode: 13 }), "Enter")).toBe(true);
  });

  it("lets an Android soft-keyboard Enter (keyCode 13) through", () => {
    expect(
      isKey(
        keydown("Unidentified", { keyCode: 229, isComposing: true }),
        "Enter",
      ),
    ).toBe(false);
    expect(isKey(keydown("Enter", { keyCode: 13 }), "Enter")).toBe(true);
  });
});

describe("ESLint keyboard selectors", () => {
  it("cover exactly the KEY_NAMES list", () => {
    const config = readFileSync(".eslintrc.json", "utf8");
    const lists = [...config.matchAll(/value=\/\^\(([^)]+)\)\$\//g)].map((m) =>
      m[1].split("|"),
    );
    expect(lists.length).toBeGreaterThan(0);
    for (const list of lists) {
      expect(new Set(list)).toEqual(new Set(KEY_NAMES));
    }
  });
});
