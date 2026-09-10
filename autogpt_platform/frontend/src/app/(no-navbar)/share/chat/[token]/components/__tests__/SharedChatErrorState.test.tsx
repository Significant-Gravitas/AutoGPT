import { afterEach, describe, expect, test, vi } from "vitest";

import { cleanup, render, screen } from "@/tests/integrations/test-utils";
import { SharedChatErrorState } from "../SharedChatErrorState";

afterEach(() => {
  cleanup();
});

describe("SharedChatErrorState", () => {
  test("renders the not-found copy", () => {
    render(<SharedChatErrorState onRetry={vi.fn()} />);
    expect(screen.getByText(/share link not found/i)).toBeDefined();
    expect(screen.getByText(/invalid or has been disabled/i)).toBeDefined();
  });

  test("invokes onRetry when the user clicks 'Try again'", () => {
    const onRetry = vi.fn();
    render(<SharedChatErrorState onRetry={onRetry} />);

    const retryButton = screen.getByRole("button", { name: /try again/i });
    retryButton.click();

    expect(onRetry).toHaveBeenCalledTimes(1);
  });
});
