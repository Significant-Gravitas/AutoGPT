import { fireEvent, render, screen } from "@/tests/integrations/test-utils";
import type { UIMessage } from "ai";
import { describe, expect, it, vi } from "vitest";
import { tickColor, tickScale, toMinimapEntries } from "../helpers";
import { ChatMinimap } from "../ChatMinimap";

function textMessage(id: string, role: "user" | "assistant", text: string) {
  return {
    id,
    role,
    parts: [{ type: "text", text }],
  } as unknown as UIMessage;
}

const MESSAGES = [
  textMessage("m1", "user", "Plan my week\nMonday first"),
  textMessage("m2", "assistant", "Here is the plan"),
  textMessage("m3", "user", "Now the budget"),
];

describe("ChatMinimap", () => {
  it("renders one tick per message once the thread is long enough", () => {
    render(<ChatMinimap messages={MESSAGES} />);
    expect(screen.getAllByRole("button")).toHaveLength(3);
    expect(screen.getByLabelText("Jump to: Plan my week")).toBeDefined();
  });

  it("renders nothing for short threads", () => {
    const { container } = render(
      <ChatMinimap messages={MESSAGES.slice(0, 2)} />,
    );
    expect(container.textContent).toBe("");
  });

  it("shows the turn's preview card on hover", () => {
    render(<ChatMinimap messages={MESSAGES} />);
    fireEvent.mouseEnter(screen.getByLabelText("Jump to: Plan my week"));
    expect(screen.getByText("Plan my week")).toBeDefined();
    expect(screen.getByText("Monday first")).toBeDefined();
  });

  it("scrolls to the message on click and on Enter", () => {
    render(
      <div>
        <ChatMinimap messages={MESSAGES} />
        <div data-message-id="m1">Plan my week</div>
      </div>,
    );
    const scrollIntoView = vi.fn();
    document
      .querySelectorAll("[data-message-id]")
      .forEach(
        (el) => ((el as HTMLElement).scrollIntoView = scrollIntoView as never),
      );

    const tick = screen.getByLabelText("Jump to: Plan my week");
    fireEvent.click(tick);
    fireEvent.keyDown(tick, { key: "Enter" });
    fireEvent.keyDown(tick, { key: " " });

    expect(scrollIntoView).toHaveBeenCalledTimes(3);
  });
});

describe("minimap helpers", () => {
  it("falls back to role labels for text-free turns", () => {
    const entries = toMinimapEntries([
      { id: "t1", role: "assistant", parts: [] } as unknown as UIMessage,
      { id: "t2", role: "user", parts: [] } as unknown as UIMessage,
    ]);
    expect(entries[0].title).toBe("Autopilot worked on this");
    expect(entries[1].title).toBe("Your message");
  });

  it("truncates long titles with an ellipsis", () => {
    const entries = toMinimapEntries([
      textMessage("t1", "user", "x".repeat(100)),
    ]);
    expect(entries[0].title.endsWith("…")).toBe(true);
    expect(entries[0].title.length).toBeLessThanOrEqual(61);
  });

  it("swells ticks toward the cursor and tapers with distance", () => {
    expect(tickScale(3, null)).toBe(0.4);
    expect(tickScale(3, 3)).toBe(1);
    expect(tickScale(4, 3)).toBeCloseTo(0.8);
    expect(tickScale(9, 3)).toBe(0.4);
    expect(tickColor(0)).toBe("bg-zinc-800");
    expect(tickColor(1)).toBe("bg-zinc-400");
    expect(tickColor(4)).toBe("bg-zinc-300");
  });
});
