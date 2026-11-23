import { cleanup, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { MessageResponse } from "../Message";

// Streamdown memoises each paragraph on its class and source position, so a
// text that changes without moving any block (a stored reply that reads
// "!" where the stream said ".") would keep the old words on screen. A
// transcript row now stays mounted when its stored copy replaces the live
// one, so the response has to notice the change itself.
describe("MessageResponse when its text is replaced", () => {
  afterEach(() => {
    cleanup();
  });

  it("shows a replacement of the same length", async () => {
    const { rerender } = render(
      <MessageResponse>You&apos;re welcome.</MessageResponse>,
    );
    await screen.findByText("You're welcome.");
    rerender(<MessageResponse>{"You're welcome!"}</MessageResponse>);
    await waitFor(() =>
      expect(screen.getByText("You're welcome!")).toBeDefined(),
    );
    expect(screen.queryByText("You're welcome.")).toBeNull();
  });

  it("keeps its paragraph node while the text only grows", async () => {
    const { rerender, container } = render(
      <MessageResponse isStreaming>Hello there</MessageResponse>,
    );
    await screen.findByText(/Hello/);
    const paragraph = container.querySelector("p");
    rerender(
      <MessageResponse isStreaming>Hello there, friend</MessageResponse>,
    );
    await waitFor(() => expect(container.textContent).toContain("friend"));
    expect(container.querySelector("p")).toBe(paragraph);
  });
});
