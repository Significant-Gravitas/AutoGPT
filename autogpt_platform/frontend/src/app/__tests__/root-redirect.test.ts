import { describe, expect, test, vi } from "vitest";

const redirect = vi.hoisted(() =>
  vi.fn(() => {
    throw new Error("NEXT_REDIRECT");
  }),
);

vi.mock("next/navigation", () => ({ redirect }));

import Page from "../page";

describe("the root page", () => {
  test("redirects to /home while rendering on the server", () => {
    expect(() => Page()).toThrow("NEXT_REDIRECT");
    expect(redirect).toHaveBeenCalledWith("/home");
  });
});
