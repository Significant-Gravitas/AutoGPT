import { fireEvent, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { ToolUIPart } from "ai";
import { render } from "@/tests/integrations/test-utils";
import { campaign } from "@/lib/openui/samples";
import { CopilotChatActionsProvider } from "../../../components/CopilotChatActionsProvider/CopilotChatActionsProvider";
import { ChainMessageParts } from "../../../components/ChatMessagesContainer/components/ChainMessageParts";
import { convertChatSessionMessagesToUiMessages } from "../../../helpers/convertChatSessionToUiMessages";

const output = {
  type: "ui_rendered",
  version: 1,
  session_id: "session-ui",
  source: campaign,
  message: "An editable campaign brief and launch checklist.",
};

function showResult(
  result: unknown = output,
  options: {
    readOnly?: boolean;
    onSend?: (message: string) => Promise<void>;
  } = {},
) {
  const part: ToolUIPart = {
    type: "tool-render_ui",
    toolCallId: "ui-call-1",
    state: "output-available",
    input: { source: campaign, summary: output.message },
    output: result,
  };
  return render(
    <CopilotChatActionsProvider
      onSend={options.onSend ?? vi.fn()}
      chatSurface={options.readOnly ? "share" : "copilot"}
    >
      <ChainMessageParts
        parts={[part]}
        messageID="ui-message"
        isCurrentlyStreaming={false}
        readOnly={options.readOnly}
      />
    </CopilotChatActionsProvider>,
  );
}

describe("OpenUI results in a Copilot conversation", () => {
  beforeEach(() => sessionStorage.clear());

  it("keeps drafts isolated when different sessions reuse a tool call ID", async () => {
    const first = showResult();
    fireEvent.change(await screen.findByLabelText("Audience"), {
      target: { value: "Private draft for the first conversation" },
    });
    first.unmount();
    showResult({ ...output, session_id: "another-session" });
    expect(
      ((await screen.findByLabelText("Audience")) as HTMLInputElement).value,
    ).toBe("Small B2B marketing teams");
  });

  it("keeps local input edits through presentation changes and remounts", async () => {
    const first = showResult();
    fireEvent.change(await screen.findByLabelText("Audience"), {
      target: { value: "Local bookshops" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Summary" }));
    expect(await screen.findByText(output.message)).toBeDefined();
    fireEvent.click(screen.getByRole("button", { name: "Explore" }));
    expect((screen.getByLabelText("Audience") as HTMLInputElement).value).toBe(
      "Local bookshops",
    );
    first.unmount();
    showResult();
    expect(
      ((await screen.findByLabelText("Audience")) as HTMLInputElement).value,
    ).toBe("Local bookshops");
  });

  it("does not send the same action twice while its first send is pending", async () => {
    let finish: () => void = () => {};
    const send = vi.fn(
      () =>
        new Promise<void>((resolve) => {
          finish = resolve;
        }),
    );
    showResult(output, { onSend: send });
    const button = await screen.findByRole("button", { name: "Build my plan" });
    fireEvent.click(button);
    fireEvent.click(button);
    expect(send).toHaveBeenCalledTimes(1);
    expect(button.hasAttribute("disabled")).toBe(true);
    finish();
    await waitFor(() => expect(button.hasAttribute("disabled")).toBe(false));
  });

  it("falls back to the summary for a future unsupported result version", async () => {
    showResult({ ...output, version: 2 });
    expect(await screen.findByText(output.message)).toBeDefined();
    expect(screen.queryByLabelText("Audience")).toBeNull();
  });
  it("renders the actual interactive result outside the collapsed tool chain", async () => {
    const send = vi.fn().mockResolvedValue(undefined);
    showResult(output, { onSend: send });
    fireEvent.change(await screen.findByRole("textbox", { name: "Audience" }), {
      target: { value: "Independent bookshops" },
    });
    fireEvent.click(screen.getByRole("button", { name: "Build my plan" }));
    await waitFor(() => expect(send).toHaveBeenCalledTimes(1));
    expect(send.mock.calls[0][0]).toContain("Independent bookshops");
    expect(send.mock.calls[0][0]).toContain("$2,500");
    expect(send.mock.calls[0][0]).toContain("An AI-powered research assistant");
    expect(send.mock.calls[0][0]).toContain("Build");
  });

  it("renders saved tool output after session hydration", async () => {
    const { messages } = convertChatSessionMessagesToUiMessages(
      "session-ui",
      [
        {
          role: "assistant",
          sequence: 1,
          tool_calls: [
            {
              id: "ui-call-1",
              type: "function",
              function: {
                name: "render_ui",
                arguments: JSON.stringify({
                  source: campaign,
                  summary: output.message,
                }),
              },
            },
          ],
        },
        {
          role: "tool",
          sequence: 2,
          tool_call_id: "ui-call-1",
          content: JSON.stringify(output),
        },
      ],
      { isComplete: true },
    );
    render(
      <CopilotChatActionsProvider onSend={vi.fn()}>
        {messages.map((message) => (
          <ChainMessageParts
            key={message.id}
            parts={message.parts}
            messageID={message.id}
            isCurrentlyStreaming={false}
          />
        ))}
      </CopilotChatActionsProvider>,
    );
    expect(
      await screen.findByRole("textbox", { name: "Audience" }),
    ).toBeDefined();
  });

  it("keeps shared results readable and prevents conversation actions", async () => {
    const send = vi.fn().mockResolvedValue(undefined);
    showResult(output, { readOnly: true, onSend: send });
    const button = await screen.findByRole("button", { name: "Build my plan" });
    expect(button.hasAttribute("disabled")).toBe(true);
    fireEvent.click(button);
    expect(send).not.toHaveBeenCalled();
  });

  it("shows the saved summary when the generated program is invalid", async () => {
    showResult({ ...output, source: 'root = UnknownWidget("broken")' });
    expect(await screen.findByText(output.message)).toBeDefined();
    expect(screen.queryByRole("textbox", { name: "Audience" })).toBeNull();
  });

  it("retains edited inputs after a failed follow-up and allows retry", async () => {
    const send = vi
      .fn()
      .mockRejectedValueOnce(new Error("offline"))
      .mockResolvedValueOnce(undefined);
    showResult(output, { onSend: send });
    const audience = await screen.findByRole("textbox", { name: "Audience" });
    fireEvent.change(audience, { target: { value: "Independent bookshops" } });
    fireEvent.click(screen.getByRole("button", { name: "Build my plan" }));
    expect((await screen.findByRole("alert")).textContent).toContain(
      "couldn't be sent",
    );
    expect((audience as HTMLInputElement).value).toBe("Independent bookshops");
    fireEvent.click(screen.getByRole("button", { name: "Build my plan" }));
    await waitFor(() => expect(send).toHaveBeenCalledTimes(2));
  });
});
