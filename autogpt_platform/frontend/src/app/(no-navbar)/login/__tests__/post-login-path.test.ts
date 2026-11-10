import { describe, expect, test } from "vitest";
import { getPostLoginPath } from "../helpers";

describe("getPostLoginPath", () => {
  test("a closed account goes to the keep-account page, even with ?next=", () => {
    expect(
      getPostLoginPath({
        nextUrl: "/copilot",
        resultNext: "/account-closed",
        rawNext: "/copilot",
      }),
    ).toBe("/account-closed");
    expect(
      getPostLoginPath({
        nextUrl: "/marketplace",
        resultNext: "/account-closed",
        rawNext: "/marketplace",
      }),
    ).toBe("/account-closed");
  });

  test("a closed account's keep-account page carries no tracking params", () => {
    expect(
      getPostLoginPath({
        nextUrl: null,
        resultNext: "/account-closed",
        rawNext: "/home?utm_source=newsletter",
      }),
    ).toBe("/account-closed");
  });

  test("an open account honours ?next= first, then the server's choice", () => {
    expect(
      getPostLoginPath({
        nextUrl: "/library",
        resultNext: "/home",
        rawNext: "/library",
      }),
    ).toBe("/library");
    expect(
      getPostLoginPath({
        nextUrl: null,
        resultNext: "/onboarding",
        rawNext: null,
      }),
    ).toBe("/onboarding");
    expect(
      getPostLoginPath({ nextUrl: null, resultNext: undefined, rawNext: null }),
    ).toBe("/");
  });

  test("the server's choice keeps a default landing's tracking params", () => {
    expect(
      getPostLoginPath({
        nextUrl: null,
        resultNext: "/onboarding",
        rawNext: "/home?utm_source=newsletter",
      }),
    ).toBe("/onboarding?utm_source=newsletter");
  });
});
