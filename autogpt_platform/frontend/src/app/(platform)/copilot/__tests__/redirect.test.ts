import { describe, expect, it, vi } from "vitest";
import Page from "../page";

vi.mock("next/navigation", () => ({
  redirect: (url: string) => {
    throw new Error(`NEXT_REDIRECT:${url}`);
  },
}));

describe("copilot redirect", () => {
  it("redirects new tasks to home", async () => {
    await expect(Page({ searchParams: Promise.resolve({}) })).rejects.toThrow(
      "NEXT_REDIRECT:/home",
    );
  });

  it("preserves chat, expert, kickoff, and repeated query parameters", async () => {
    await expect(
      Page({
        searchParams: Promise.resolve({
          sessionId: "session/one",
          expertId: "expert-one",
          kickoff: "1",
          tag: ["a", "b"],
          absent: undefined,
        }),
      }),
    ).rejects.toThrow(
      "NEXT_REDIRECT:/home?sessionId=session%2Fone&expertId=expert-one&kickoff=1&tag=a&tag=b",
    );
  });
});
