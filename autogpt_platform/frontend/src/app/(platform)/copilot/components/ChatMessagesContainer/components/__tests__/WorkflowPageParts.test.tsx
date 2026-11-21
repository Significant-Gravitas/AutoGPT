import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import output from "../../../../tools/DesignWorkflowPage/__tests__/workflowPageOutput.json";
import {
  APPLY_MESSAGE,
  DISCARD_MESSAGE,
} from "../../../../tools/DesignWorkflowPage/helpers";
import { CopilotChatActionsProvider } from "../../../CopilotChatActionsProvider/CopilotChatActionsProvider";
import type { MessagePart } from "../../helpers";
import { ChainMessageParts } from "../ChainMessageParts";

vi.mock("../../../ToolChain/ToolChain", () => ({
  ToolChain: () => <div data-testid="tool-chain" />,
}));

vi.mock("next/navigation", async (importOriginal) => ({
  ...(await importOriginal<typeof import("next/navigation")>()),
  useRouter: () => ({ push: vi.fn() }),
}));

function pagePart(overrides: Record<string, unknown> = {}) {
  return {
    type: "tool-design_workflow_page",
    toolCallId: "call-1",
    state: "output-available",
    input: {},
    output,
    ...overrides,
  } as unknown as MessagePart;
}

function renderParts(parts: MessagePart[], canActOnCards = true) {
  const onSend = vi.fn();
  render(
    <CopilotChatActionsProvider onSend={onSend}>
      <ChainMessageParts
        parts={parts}
        messageID="message-1"
        isCurrentlyStreaming={false}
        canActOnCards={canActOnCards}
      />
    </CopilotChatActionsProvider>,
  );
  return { onSend };
}

describe("workflow page preview card", () => {
  afterEach(cleanup);

  it("draws the page the backend resolved, outside the tool chain", () => {
    renderParts([pagePart()]);

    expect(screen.getByText("Workflow page preview")).toBeDefined();
    expect(screen.getByText("Lead research")).toBeDefined();
    expect(screen.getByText("Leads")).toBeDefined();
    expect(screen.getByText("Ada")).toBeDefined();
    expect(screen.getByText("VP Sales")).toBeDefined();
    expect(screen.queryByTestId("tool-chain")).toBeNull();
  });

  it.each([
    ["Apply", APPLY_MESSAGE],
    ["Discard", DISCARD_MESSAGE],
  ])("sends %s as the user's reply, once", (label, message) => {
    const { onSend } = renderParts([pagePart()]);

    fireEvent.click(screen.getByRole("button", { name: label }));
    fireEvent.click(screen.getByRole("button", { name: label }));

    expect(onSend).toHaveBeenCalledTimes(1);
    expect(onSend).toHaveBeenCalledWith(message);
  });

  it("offers no buttons on a page that was already saved", () => {
    renderParts([pagePart({ output: { ...output, saved: true } })]);

    expect(screen.getByText("Workflow page saved")).toBeDefined();
    expect(screen.queryByRole("button", { name: "Apply" })).toBeNull();
    expect(
      screen.getByRole("link", { name: "Open it in the library" }),
    ).toBeDefined();
  });
});
