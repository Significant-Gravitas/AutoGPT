import { fireEvent, screen } from "@testing-library/react";
import type { ToolUIPart } from "ai";
import { StrictMode } from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { render } from "@/tests/integrations/test-utils";
import { CopilotChatActionsProvider } from "../../../components/CopilotChatActionsProvider/CopilotChatActionsProvider";
import { ChainMessageParts } from "../../../components/ChatMessagesContainer/components/ChainMessageParts";

const updatedPlan = `root = Workspace("Moving", "Progress you reported", [tasks])
tasks = Checklist("Before moving day", [
  {"title":"Book movers","detail":"Booking confirmed by you","done":true},
  {"title":"Buy boxes","detail":"Already purchased","done":true},
  {"title":"Pack books","detail":"Use the boxes you bought","done":false}
])`;

function showPlan(source = updatedPlan, readOnly = false) {
  const send = vi.fn();
  const part: ToolUIPart = {
    type: "tool-render_ui",
    toolCallId: "moving-progress",
    state: "output-available",
    input: {},
    output: {
      type: "ui_rendered",
      version: 1,
      session_id: "moving-session",
      source,
      message: "Moving progress",
    },
  };
  const view = render(
    <StrictMode>
      <CopilotChatActionsProvider
        onSend={send}
        chatSurface={readOnly ? "share" : "copilot"}
      >
        <ChainMessageParts
          parts={[part]}
          messageID="moving-message"
          isCurrentlyStreaming={false}
          readOnly={readOnly}
        />
      </CopilotChatActionsProvider>
    </StrictMode>,
  );
  return { ...view, send };
}

describe("reported checklist progress in chat", () => {
  beforeEach(() => sessionStorage.clear());

  it("starts with reported tasks checked and restores local corrections", async () => {
    const first = showPlan();
    const movers = (await screen.findByRole("checkbox", {
      name: /Book movers/,
    })) as HTMLInputElement;
    expect(movers.checked).toBe(true);
    expect(screen.getByText("2 of 3 complete")).toBeDefined();
    fireEvent.click(movers);
    fireEvent.click(screen.getByRole("checkbox", { name: /Pack books/ }));
    expect(first.send).not.toHaveBeenCalled();
    first.unmount();
    showPlan();
    expect(
      (
        (await screen.findByRole("checkbox", {
          name: /Book movers/,
        })) as HTMLInputElement
      ).checked,
    ).toBe(false);
    expect(
      (screen.getByRole("checkbox", { name: /Pack books/ }) as HTMLInputElement)
        .checked,
    ).toBe(true);
    expect(screen.getByText("2 of 3 complete")).toBeDefined();
  });

  it("continues to support saved checklists without initial progress", async () => {
    showPlan(updatedPlan.replace(/,"done":(?:true|false)/g, ""));
    const boxes = (await screen.findByRole("checkbox", {
      name: /Buy boxes/,
    })) as HTMLInputElement;
    expect(boxes.checked).toBe(false);
    fireEvent.click(boxes);
    expect(screen.getByText("1 of 3 complete")).toBeDefined();
  });

  it("preserves an explicitly cleared checklist after reopening", async () => {
    const first = showPlan();
    const movers = (await screen.findByRole("checkbox", {
      name: /Book movers/,
    })) as HTMLInputElement;
    expect(movers.checked).toBe(true);
    fireEvent.click(movers);
    fireEvent.click(screen.getByRole("checkbox", { name: /Buy boxes/ }));
    expect(screen.getByText("0 of 3 complete")).toBeDefined();
    first.unmount();
    showPlan();
    expect(await screen.findByText("0 of 3 complete")).toBeDefined();
  });

  it("shows reported progress in shared views without allowing changes", async () => {
    showPlan(updatedPlan, true);
    const movers = (await screen.findByRole("checkbox", {
      name: /Book movers/,
    })) as HTMLInputElement;
    expect(movers.checked).toBe(true);
    expect(movers.disabled).toBe(true);
    fireEvent.click(movers);
    expect(screen.getByText("2 of 3 complete")).toBeDefined();
  });
});
