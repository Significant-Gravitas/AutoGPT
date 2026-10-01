import { afterEach, describe, expect, it } from "vitest";

import {
  LAYOUT_HINT_COOKIE,
  parseLayoutHint,
  persistLayoutHint,
  readLayoutHint,
  resolveNewLayout,
} from "../helpers";

const session = { name: "better-auth.session_token", value: "token" };
const secureSession = {
  name: "__Secure-better-auth.session_token",
  value: "token",
};

afterEach(() => {
  document.cookie = `${LAYOUT_HINT_COOKIE}=; path=/; max-age=0`;
});

describe("parseLayoutHint", () => {
  it("accepts only the two known shells", () => {
    expect(parseLayoutHint("new")).toBe("new");
    expect(parseLayoutHint("classic")).toBe("classic");
    expect(parseLayoutHint("sidebar")).toBeUndefined();
    expect(parseLayoutHint("")).toBeUndefined();
    expect(parseLayoutHint(undefined)).toBeUndefined();
  });
});

describe("readLayoutHint", () => {
  it("reads the hint alongside a session cookie", () => {
    expect(
      readLayoutHint([session, { name: LAYOUT_HINT_COOKIE, value: "new" }]),
    ).toBe("new");
    expect(
      readLayoutHint([
        { name: LAYOUT_HINT_COOKIE, value: "classic" },
        secureSession,
      ]),
    ).toBe("classic");
  });

  it("ignores the hint for a visitor without a session", () => {
    expect(readLayoutHint([{ name: LAYOUT_HINT_COOKIE, value: "new" }])).toBe(
      undefined,
    );
    expect(readLayoutHint([])).toBeUndefined();
  });

  it("ignores a hint it does not recognise", () => {
    expect(
      readLayoutHint([session, { name: LAYOUT_HINT_COOKIE, value: "nope" }]),
    ).toBeUndefined();
    expect(readLayoutHint([session])).toBeUndefined();
  });
});

describe("persistLayoutHint", () => {
  it("writes the shell to the layout cookie", () => {
    persistLayoutHint(true);
    expect(document.cookie).toContain(`${LAYOUT_HINT_COOKIE}=new`);
    persistLayoutHint(false);
    expect(document.cookie).toContain(`${LAYOUT_HINT_COOKIE}=classic`);
    expect(document.cookie).not.toContain(`${LAYOUT_HINT_COOKIE}=new`);
  });
});

describe("resolveNewLayout", () => {
  it("is undecided with no answer and no hint", () => {
    expect(
      resolveNewLayout({
        enabled: false,
        answered: false,
        ready: false,
        hint: undefined,
      }),
    ).toBeUndefined();
  });

  it("follows the hint until the vendor answers", () => {
    expect(
      resolveNewLayout({
        enabled: false,
        answered: false,
        ready: false,
        hint: "new",
      }),
    ).toBe(true);
    expect(
      resolveNewLayout({
        enabled: true,
        answered: false,
        ready: false,
        hint: "classic",
      }),
    ).toBe(false);
  });

  it("lets the vendor's answer beat the hint", () => {
    expect(
      resolveNewLayout({
        enabled: false,
        answered: true,
        ready: true,
        hint: "new",
      }),
    ).toBe(false);
    expect(
      resolveNewLayout({
        enabled: true,
        answered: true,
        ready: true,
        hint: "classic",
      }),
    ).toBe(true);
  });

  it("keeps the hint over a timeout fallback, and uses the fallback without one", () => {
    expect(
      resolveNewLayout({
        enabled: false,
        answered: false,
        ready: true,
        hint: "new",
      }),
    ).toBe(true);
    expect(
      resolveNewLayout({
        enabled: false,
        answered: false,
        ready: true,
        hint: undefined,
      }),
    ).toBe(false);
  });
});
