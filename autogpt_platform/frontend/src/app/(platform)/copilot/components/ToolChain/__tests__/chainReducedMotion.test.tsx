import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  cleanup,
  render as baseRender,
  screen,
} from "@/tests/integrations/test-utils";
import { CopilotChatActionsProvider } from "../../CopilotChatActionsProvider/CopilotChatActionsProvider";
import type { MessagePart } from "../../ChatMessagesContainer/helpers";
import { SwapText } from "../SwapText";
import { ToolChain } from "../ToolChain";

vi.mock("framer-motion", async (importOriginal) => ({
  ...(await importOriginal<typeof import("framer-motion")>()),
  useReducedMotion: () => true,
}));

function search(state: string) {
  return {
    type: "tool-web_search",
    state,
    toolCallId: "call-search",
    input: { query: "hotels" },
    output: state === "output-available" ? { results: [1] } : undefined,
  } as MessagePart;
}

function chain(parts: MessagePart[]) {
  return (
    <CopilotChatActionsProvider onSend={vi.fn()}>
      <ToolChain parts={parts} isStreaming />
    </CopilotChatActionsProvider>
  );
}

describe("ToolChain under reduced motion", () => {
  beforeEach(() => {
    vi.useFakeTimers({ shouldAdvanceTime: false });
  });
  afterEach(() => {
    cleanup();
    vi.useRealTimers();
  });

  it("shows a fast call finished at once, with no entrance", () => {
    const { container, rerender } = baseRender(
      chain([search("input-available")]),
    );
    const header = screen.getAllByRole("button", {
      name: 'Searching the web for "hotels"…',
    })[0];
    expect(header.parentElement?.style.opacity ?? "").not.toBe("0");

    rerender(chain([search("output-available")]));

    expect(
      container.querySelector("[data-state]")?.getAttribute("data-state"),
    ).toBe("done");
    expect(screen.getByText('Searched the web for "hotels"')).toBeDefined();
  });

  it("changes a held label at once", () => {
    const { rerender } = baseRender(<SwapText text="One" minHoldMs={500} />);
    rerender(<SwapText text="Two" minHoldMs={500} />);

    expect(screen.getByText("Two")).toBeDefined();
  });
});
