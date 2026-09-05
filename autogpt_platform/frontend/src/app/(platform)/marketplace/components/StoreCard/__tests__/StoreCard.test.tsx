import { createEvent, fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { StoreCard } from "../StoreCard";

function renderStoreCard(onClick = vi.fn()) {
  render(
    <StoreCard
      agentName="Research workflow"
      agentImage=""
      description="Researches a topic"
      runs={10}
      rating={4.5}
      onClick={onClick}
      avatarSrc=""
    />,
  );

  return { card: screen.getByRole("button"), onClick };
}

describe("StoreCard keyboard handling", () => {
  it("does not activate while an IME is composing", () => {
    const { card, onClick } = renderStoreCard();

    fireEvent.keyDown(card, { key: "Enter", isComposing: true });
    fireEvent.keyDown(card, { key: " ", isComposing: true });

    expect(onClick).not.toHaveBeenCalled();
  });

  it.each(["Enter", " "])("activates on %s without scrolling", (key) => {
    const { card, onClick } = renderStoreCard();
    const event = createEvent.keyDown(card, { key });

    fireEvent(card, event);

    expect(onClick).toHaveBeenCalledTimes(1);
    expect(event.defaultPrevented).toBe(true);
  });
});
