import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { UserMessageClamp } from "../UserMessageClamp";

function setMeasuredHeights(args: {
  scrollHeight: number;
  clientHeight: number;
}) {
  Object.defineProperty(HTMLElement.prototype, "scrollHeight", {
    configurable: true,
    get: () => args.scrollHeight,
  });
  Object.defineProperty(HTMLElement.prototype, "clientHeight", {
    configurable: true,
    get: () => args.clientHeight,
  });
}

afterEach(() => {
  cleanup();
  delete (HTMLElement.prototype as unknown as Record<string, unknown>)
    .scrollHeight;
  delete (HTMLElement.prototype as unknown as Record<string, unknown>)
    .clientHeight;
});

describe("UserMessageClamp", () => {
  it("shows the clamped content without a Read more button when it fits", () => {
    setMeasuredHeights({ scrollHeight: 100, clientHeight: 100 });

    render(<UserMessageClamp>Short message</UserMessageClamp>);

    expect(screen.getByText("Short message")).toBeDefined();
    expect(screen.getByText("Short message").className).toContain(
      "line-clamp-6",
    );
    expect(screen.queryByRole("button")).toBeNull();
  });

  it("expands and collapses overflowing content via Read more / Show less", () => {
    setMeasuredHeights({ scrollHeight: 300, clientHeight: 120 });

    render(<UserMessageClamp>Very long message</UserMessageClamp>);

    const content = screen.getByText("Very long message");
    expect(content.className).toContain("line-clamp-6");

    fireEvent.click(screen.getByRole("button", { name: "Read more" }));
    expect(content.className).not.toContain("line-clamp-6");

    fireEvent.click(screen.getByRole("button", { name: "Show less" }));
    expect(content.className).toContain("line-clamp-6");
    expect(screen.getByRole("button", { name: "Read more" })).toBeDefined();
  });
});
