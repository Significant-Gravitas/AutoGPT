import { afterEach, describe, expect, test, vi } from "vitest";

import {
  cleanup,
  fireEvent,
  render,
  screen,
} from "@/tests/integrations/test-utils";
import { SharedChatErrorState } from "../SharedChatErrorState";

afterEach(() => {
  cleanup();
});

describe("SharedChatErrorState", () => {
  test("says the link is dead, with no retry, for a dead link", () => {
    render(<SharedChatErrorState isDeadLink onRetry={vi.fn()} />);

    expect(
      screen.getByRole("heading", { name: "Share link not found" }),
    ).toBeDefined();
    expect(screen.getByText(/invalid or has been disabled/i)).toBeDefined();
    expect(screen.queryByRole("button", { name: "Try again" })).toBeNull();
  });

  test("offers 'Try again', without calling the link dead, for any other failure", () => {
    const onRetry = vi.fn();
    render(<SharedChatErrorState isDeadLink={false} onRetry={onRetry} />);

    expect(
      screen.getByRole("heading", { name: "Unable to load" }),
    ).toBeDefined();
    expect(screen.queryByText(/share link not found/i)).toBeNull();
    expect(screen.queryByText(/disabled by the owner/i)).toBeNull();

    fireEvent.click(screen.getByRole("button", { name: "Try again" }));

    expect(onRetry).toHaveBeenCalledTimes(1);
  });
});
