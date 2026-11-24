import { render, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { MessageResponse, preloadMessageResponse } from "../Message";

// The shared setup hands tests the renderer loaded; this file sees the real wait.
vi.unmock("@/components/organisms/Message/Message");

const BOLD = '[data-streamdown="strong"]';

describe("MessageResponse before its renderer has loaded", () => {
  it("shows the reply's text, then formats it once the renderer is here", async () => {
    const { container } = render(
      <MessageResponse>{"The **harbour** opens at dawn."}</MessageResponse>,
    );

    expect(container.textContent).toBe("The **harbour** opens at dawn.");
    expect(container.querySelector(BOLD)).toBeNull();

    await preloadMessageResponse();
    await waitFor(() =>
      expect(container.querySelector(BOLD)?.textContent).toBe("harbour"),
    );
    expect(container.textContent).toBe("The harbour opens at dawn.");
  });
});
