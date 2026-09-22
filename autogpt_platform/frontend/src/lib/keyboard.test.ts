import { describe, expect, it } from "vitest";
import { isComposingEvent, isKey, isKeyIgnoringComposition } from "./keyboard";

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

  it("guards the candidate-window paging keys too", () => {
    expect(isKey(keydown("PageDown"), "PageDown")).toBe(true);
    expect(isKey(keydown("PageDown", { isComposing: true }), "PageDown")).toBe(
      false,
    );
    expect(isKey(keydown("Home", { isComposing: true }), "Home")).toBe(false);
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

describe("isKeyIgnoringComposition", () => {
  it("matches the key name whether or not an IME is composing", () => {
    expect(isKeyIgnoringComposition(keydown("Tab"), "Tab")).toBe(true);
    expect(
      isKeyIgnoringComposition(keydown("Tab", { isComposing: true }), "Tab"),
    ).toBe(true);
    expect(
      isKeyIgnoringComposition(keydown("Tab", { keyCode: 229 }), "Tab"),
    ).toBe(true);
  });

  it("still does not match a different key", () => {
    expect(
      isKeyIgnoringComposition(keydown("Enter", { isComposing: true }), "Tab"),
    ).toBe(false);
  });
});
