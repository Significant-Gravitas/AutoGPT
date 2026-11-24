import { describe, expect, it, vi } from "vitest";

const { preloadMessageResponse } = vi.hoisted(() => ({
  preloadMessageResponse: vi.fn(() => Promise.resolve()),
}));

vi.mock("@/components/organisms/Message/Message", async (importActual) => ({
  ...(await importActual<
    typeof import("@/components/organisms/Message/Message")
  >()),
  preloadMessageResponse,
}));

describe("the chat page's code", () => {
  // An open chat's messages arrive after hydration; the renderer they need
  // downloads during it rather than after them.
  it("starts loading the message renderer as soon as it runs", async () => {
    await import("../CopilotPage");

    expect(preloadMessageResponse).toHaveBeenCalledTimes(1);
  });
});
