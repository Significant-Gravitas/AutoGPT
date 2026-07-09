import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";

import {
  cleanup,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import { ShareActions } from "../ShareActions";

const mockUseSupabase = vi.hoisted(() => vi.fn());

vi.mock("@/lib/supabase/hooks/useSupabase", () => ({
  useSupabase: mockUseSupabase,
}));

// Clipboard isn't available in jsdom by default; defineProperty so
// the copy plumbing can be exercised without crashing.
const clipboardWrite = vi.fn(async (_text: string) => {});

beforeEach(() => {
  Object.defineProperty(navigator, "clipboard", {
    configurable: true,
    value: { writeText: clipboardWrite },
  });
  Object.defineProperty(window, "location", {
    configurable: true,
    value: new URL("http://localhost/share/chat/some-token"),
  });
  clipboardWrite.mockClear();
  mockUseSupabase.mockReset();
});

afterEach(() => {
  cleanup();
});

describe("ShareActions", () => {
  test("renders Sign up CTA when the viewer is anonymous", () => {
    mockUseSupabase.mockReturnValue({
      isLoggedIn: false,
      isUserLoading: false,
    });
    render(<ShareActions />);

    expect(screen.getByRole("button", { name: /copy link/i })).toBeDefined();
    const signUp = screen.getByRole("link", { name: /sign up/i });
    expect(signUp.getAttribute("href")).toBe("/signup");
  });

  test("hides Sign up CTA when the viewer is signed in", () => {
    mockUseSupabase.mockReturnValue({
      isLoggedIn: true,
      isUserLoading: false,
    });
    render(<ShareActions />);

    expect(screen.getByRole("button", { name: /copy link/i })).toBeDefined();
    expect(screen.queryByRole("link", { name: /sign up/i })).toBeNull();
  });

  test("hides Sign up CTA while auth state is still loading", () => {
    // First paint after navigation: useSupabase hasn't initialised yet.
    // Showing then hiding the CTA on hydration looks broken to a
    // signed-in viewer opening their own share, so we suppress it.
    mockUseSupabase.mockReturnValue({
      isLoggedIn: false,
      isUserLoading: true,
    });
    render(<ShareActions />);

    expect(screen.queryByRole("link", { name: /sign up/i })).toBeNull();
    // Copy link remains available — anonymous and signed-in alike
    // benefit from the one-click handoff.
    expect(screen.getByRole("button", { name: /copy link/i })).toBeDefined();
  });

  test("clicking Copy link writes the current URL to the clipboard", async () => {
    mockUseSupabase.mockReturnValue({
      isLoggedIn: false,
      isUserLoading: false,
    });
    render(<ShareActions />);

    const copy = screen.getByRole("button", { name: /copy link/i });
    copy.click();

    await waitFor(() => {
      expect(clipboardWrite).toHaveBeenCalledTimes(1);
    });
    expect(clipboardWrite.mock.calls[0][0]).toBe(
      "http://localhost/share/chat/some-token",
    );
    // Button briefly flips to "Copied" — pinning the setCopied(true)
    // branch in handleCopy.
    await screen.findByRole("button", { name: /copied/i });
  });

  test("clipboard failure leaves the button on 'Copy link' and does not flash 'Copied'", async () => {
    mockUseSupabase.mockReturnValue({
      isLoggedIn: false,
      isUserLoading: false,
    });
    clipboardWrite.mockRejectedValueOnce(new Error("nope"));
    render(<ShareActions />);

    const copy = screen.getByRole("button", { name: /copy link/i });
    copy.click();

    await waitFor(() => {
      expect(clipboardWrite).toHaveBeenCalled();
    });
    // The catch path swallows the rejection — the "Copied" affordance
    // must NOT appear because setCopied is gated inside the try block.
    expect(screen.queryByRole("button", { name: /copied/i })).toBeNull();
    expect(screen.getByRole("button", { name: /copy link/i })).toBeDefined();
  });
});
