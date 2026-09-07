import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  act,
  cleanup,
  render as baseRender,
  waitFor,
} from "@/tests/integrations/test-utils";
import { useCopilotUIStore } from "@/app/(platform)/copilot/store";
import { CopilotChatActionsProvider } from "../../CopilotChatActionsProvider/CopilotChatActionsProvider";
import type { MessagePart } from "../../ChatMessagesContainer/helpers";
import { ToolChain } from "../ToolChain";

const capture = vi.hoisted(() => vi.fn());
vi.mock("posthog-js", () => ({ default: { capture } }));

// What the stubbed card registers with the chain — the two fields the chain's
// terminal-state detection reads.
const card = vi.hoisted(() => ({
  current: { ready: false, justConnected: false },
}));

vi.mock("../../SetupRequirementsCard/SetupRequirementsCard", async () => {
  const { useContext, useEffect } = await import("react");
  const { ChainActionsContext } = await import("../chainActions");

  function SetupRequirementsCard() {
    const chainActions = useContext(ChainActionsContext);
    const { ready, justConnected } = card.current;
    useEffect(() => {
      if (!chainActions) return;
      chainActions.register({
        id: "github",
        ready,
        justConnected,
        buildMessage: () => "I've configured the required credentials.",
        connectors: {
          id: "github",
          fields: [],
          selected: {},
          onChange: () => {},
          onConnected: () => {},
        },
      });
      return () => chainActions.unregister("github");
    }, [chainActions, ready, justConnected]);
    return <div>setup-card</div>;
  }

  return { SetupRequirementsCard };
});

function setupPart(): MessagePart {
  return {
    type: "tool-connect_integration",
    state: "output-available",
    toolCallId: "call-connect_integration",
    input: {},
    output: {
      type: "setup_requirements",
      setup_info: { agent_name: "GitHub" },
    },
  } as MessagePart;
}

function render(ui: React.ReactElement) {
  const { rerender, ...rest } = baseRender(
    <CopilotChatActionsProvider onSend={vi.fn()}>
      {ui}
    </CopilotChatActionsProvider>,
  );
  return {
    ...rest,
    rerender: (next: React.ReactElement) =>
      rerender(
        <CopilotChatActionsProvider onSend={vi.fn()}>
          {next}
        </CopilotChatActionsProvider>,
      ),
  };
}

function capturedFailures() {
  return capture.mock.calls.filter(([event]) =>
    String(event).startsWith("credential_"),
  );
}

beforeEach(() => {
  vi.clearAllMocks();
  card.current = { ready: false, justConnected: false };
});

afterEach(() => {
  cleanup();
  vi.useRealTimers();
  useCopilotUIStore.setState({ initialPrompt: null, sentMessageCount: 0 });
});

describe("a card that signed in and never became ready", () => {
  it("counts the stuck card once the connect settles", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    card.current = { ready: false, justConnected: true };

    render(<ToolChain parts={[setupPart()]} isStreaming={false} />);
    await act(async () => {
      vi.advanceTimersByTime(6000);
    });

    expect(capture).toHaveBeenCalledWith(
      "credential_proceed_stuck_after_connect",
      { failure_class: "class_11_credential_not_wired_to_card" },
    );
  });

  it("counts nothing when the card becomes ready inside the settle window", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    card.current = { ready: false, justConnected: true };

    const { rerender } = render(
      <ToolChain parts={[setupPart()]} isStreaming={false} />,
    );
    await act(async () => {
      vi.advanceTimersByTime(1000);
    });

    card.current = { ready: true, justConnected: true };
    rerender(<ToolChain parts={[setupPart()]} isStreaming={false} />);
    await act(async () => {
      vi.advanceTimersByTime(10000);
    });

    expect(capturedFailures()).toHaveLength(0);
  });
});

describe("a Proceed restored from chat history", () => {
  it("counts a chain that offers Proceed having never streamed or connected", async () => {
    card.current = { ready: true, justConnected: false };

    render(<ToolChain parts={[setupPart()]} isStreaming={false} />);

    await waitFor(() =>
      expect(capture).toHaveBeenCalledWith(
        "credential_proceed_stale_from_history",
        { failure_class: "class_13_chain_turn_mismatch" },
      ),
    );
  });

  it("counts nothing for a chain that streamed in this page life", async () => {
    card.current = { ready: true, justConnected: false };

    const { rerender } = render(
      <ToolChain parts={[setupPart()]} isStreaming={true} />,
    );
    await act(async () => {});
    rerender(<ToolChain parts={[setupPart()]} isStreaming={false} />);
    await act(async () => {});

    expect(capturedFailures()).toHaveLength(0);
  });
});
