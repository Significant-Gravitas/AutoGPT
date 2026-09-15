import { act, render } from "@testing-library/react";
import { StrictMode } from "react";
import { describe, expect, it, vi } from "vitest";
import TallyPopupSimple from "../TallyPopup";

vi.mock("@/lib/auth/actions", () => ({
  getCurrentUser: vi.fn().mockResolvedValue({ user: null }),
}));

vi.mock("@sentry/nextjs", () => ({
  getReplay: vi.fn(),
}));

const scriptSelector = 'script[src="https://tally.so/widgets/embed.js"]';

describe("Tally popup cleanup", () => {
  it("removes its script on unmount after StrictMode remounts", async () => {
    const { unmount } = render(
      <StrictMode>
        <TallyPopupSimple />
      </StrictMode>,
    );
    await act(async () => {});

    expect(document.head.querySelectorAll(scriptSelector)).toHaveLength(1);
    unmount();
    expect(document.head.querySelector(scriptSelector)).toBeNull();
  });

  it("still removes the message listener when its script is already detached", async () => {
    const removeListener = vi.spyOn(window, "removeEventListener");
    const { unmount } = render(<TallyPopupSimple />);
    await act(async () => {});
    const script = document.head.querySelector(scriptSelector);
    expect(script).not.toBeNull();
    script?.remove();

    expect(unmount).not.toThrow();
    expect(removeListener).toHaveBeenCalledWith(
      "message",
      expect.any(Function),
    );
  });
});
