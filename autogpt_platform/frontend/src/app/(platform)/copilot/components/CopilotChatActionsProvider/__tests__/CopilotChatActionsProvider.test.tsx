import { render } from "@testing-library/react";
import { memo } from "react";
import { describe, expect, it, vi } from "vitest";
import { CopilotChatActionsProvider } from "../CopilotChatActionsProvider";
import { useCopilotChatActions } from "../useCopilotChatActions";

type Actions = ReturnType<typeof useCopilotChatActions>;

function setup() {
  const seen: Actions[] = [];
  const Card = memo(function Card() {
    seen.push(useCopilotChatActions());
    return null;
  });
  return { seen, card: <Card /> };
}

describe("CopilotChatActionsProvider", () => {
  it("keeps the actions when the page re-renders with a new onBackendTurn", () => {
    const onSend = vi.fn();
    const { seen, card } = setup();
    const view = render(
      <CopilotChatActionsProvider
        onSend={onSend}
        onBackendTurn={() => undefined}
        sessionId="s1"
        assistantName="Otto"
      >
        {card}
      </CopilotChatActionsProvider>,
    );

    const latest = vi.fn();
    view.rerender(
      <CopilotChatActionsProvider
        onSend={onSend}
        onBackendTurn={latest}
        sessionId="s1"
        assistantName="Otto"
      >
        {card}
      </CopilotChatActionsProvider>,
    );

    expect(seen).toHaveLength(1);
    seen[0].onBackendTurn?.();
    expect(latest).toHaveBeenCalledTimes(1);
  });

  it("publishes new actions when a field changes", () => {
    const { seen, card } = setup();
    const firstSend = vi.fn();
    const view = render(
      <CopilotChatActionsProvider onSend={firstSend} sessionId="s1">
        {card}
      </CopilotChatActionsProvider>,
    );

    const nextSend = vi.fn();
    view.rerender(
      <CopilotChatActionsProvider onSend={nextSend} sessionId="s2">
        {card}
      </CopilotChatActionsProvider>,
    );

    expect(seen).toHaveLength(2);
    expect(seen[1]).toMatchObject({
      onSend: nextSend,
      sessionId: "s2",
      chatSurface: "copilot",
    });
  });

  it("leaves onBackendTurn absent when the surface has none", () => {
    const { seen, card } = setup();
    render(
      <CopilotChatActionsProvider onSend={vi.fn()} chatSurface="share">
        {card}
      </CopilotChatActionsProvider>,
    );

    expect(seen[0].onBackendTurn).toBeUndefined();
    expect(seen[0].chatSurface).toBe("share");
  });
});
