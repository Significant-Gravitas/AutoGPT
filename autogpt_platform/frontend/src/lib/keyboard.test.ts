import { describe, expect, it } from "vitest";
import { isComposingEvent, isKey } from "./keyboard";

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

  it("does not match a different key", () => {
    expect(isKey(keydown("Escape"), "Enter")).toBe(false);
  });

  it("never matches while composing, even if the key name matches", () => {
    expect(isKey(keydown("Enter", { isComposing: true }), "Enter")).toBe(false);
    expect(isKey(reactKeydown("Enter", { keyCode: 229 }), "Enter")).toBe(false);
  });
});
