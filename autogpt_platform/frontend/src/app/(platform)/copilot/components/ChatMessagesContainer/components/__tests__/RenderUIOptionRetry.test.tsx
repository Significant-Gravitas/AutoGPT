import { Toaster } from "@/components/molecules/Toast/toaster";
import {
  act,
  cleanup,
  fireEvent,
  render,
  screen,
} from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import output from "../../../../tools/RenderUI/__tests__/renderUIOutput.json";
import { CopilotChatActionsProvider } from "../../../CopilotChatActionsProvider/CopilotChatActionsProvider";
import type { MessagePart } from "../../helpers";
import { ChainMessageParts } from "../ChainMessageParts";

vi.mock("../../../ToolChain/ToolChain", () => ({
  ToolChain: () => <div data-testid="tool-chain" />,
}));

vi.mock("@/services/analytics/posthog-capture", () => ({
  capturePostHogEvent: vi.fn(),
}));

afterEach(cleanup);

it("lets the user retry a visual option after its send fails", async () => {
  const firstSend = Promise.withResolvers<void>();
  const onSend = vi
    .fn<(message: string) => Promise<void>>()
    .mockReturnValueOnce(firstSend.promise)
    .mockResolvedValueOnce(undefined);
  const part = {
    type: "tool-render_ui",
    toolCallId: "call-1",
    state: "output-available",
    input: {},
    output: {
      ...output,
      blocks: output.blocks.filter((block) => block.type === "options"),
    },
  } as unknown as MessagePart;

  render(
    <>
      <Toaster />
      <CopilotChatActionsProvider onSend={onSend}>
        <ChainMessageParts
          parts={[part]}
          messageID="message-1"
          isCurrentlyStreaming={false}
          canActOnCards
        />
      </CopilotChatActionsProvider>
    </>,
  );

  const option = screen.getByRole<HTMLButtonElement>("button", {
    name: /^Fix enrichment/,
  });
  fireEvent.click(option);
  expect(option.disabled).toBe(true);
  fireEvent.click(option);
  expect(onSend).toHaveBeenCalledExactlyOnceWith("Fix enrichment");

  await act(async () => {
    firstSend.reject(new Error("Request failed"));
  });

  expect(
    await screen.findByText("Couldn't confirm the message was sent"),
  ).toBeDefined();
  expect(
    screen.getByText(
      /Request failed.*check the chat before selecting the same option again/i,
    ),
  ).toBeDefined();
  expect(option.disabled).toBe(false);
  expect(onSend).toHaveBeenCalledTimes(1);

  await act(async () => {
    fireEvent.click(option);
  });

  expect(onSend).toHaveBeenNthCalledWith(2, "Fix enrichment");
  expect(option.disabled).toBe(true);
  fireEvent.click(option);
  expect(onSend).toHaveBeenCalledTimes(2);
});
