import { Toaster } from "@/components/molecules/Toast/toaster";
import { ApiError } from "@/lib/autogpt-server-api/helpers";
import {
  act,
  cleanup,
  fireEvent,
  render,
  screen,
} from "@testing-library/react";
import { afterEach, expect, it, vi } from "vitest";
import { CopilotAuthError } from "../../../../helpers";
import { QueueFollowUpFullError } from "../../../../helpers/queueFollowUpMessage";
import { SendInFlightError } from "../../../../stream/turnRuntime";
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

vi.mock("@/hooks/useIsBillingEnabled", () => ({
  useIsBillingEnabled: () => true,
}));

afterEach(cleanup);

const optionPart = {
  type: "tool-render_ui",
  toolCallId: "call-1",
  state: "output-available",
  input: {},
  output: {
    ...output,
    blocks: output.blocks.filter((block) => block.type === "options"),
  },
} as unknown as MessagePart;

function renderOptions(
  onSend: (message: string) => Promise<void>,
  hasLaterMessage = false,
) {
  return (
    <>
      <Toaster />
      <CopilotChatActionsProvider onSend={onSend}>
        <ChainMessageParts
          parts={[optionPart]}
          messageID="message-1"
          isCurrentlyStreaming={false}
          canActOnCards={!hasLaterMessage}
        />
        {hasLaterMessage && (
          <ChainMessageParts
            parts={[{ ...optionPart, toolCallId: "call-2" } as MessagePart]}
            messageID="message-2"
            isCurrentlyStreaming={false}
            canActOnCards
          />
        )}
      </CopilotChatActionsProvider>
    </>
  );
}

it.each([
  new TypeError("Failed to fetch"),
  new Error("Expected 200 from /messages/pending; got 429"),
  new Error("This conversation is temporarily unavailable; try again shortly"),
  undefined,
])(
  "keeps an unconfirmed option from sending again without blocking a later card (%s)",
  async (error) => {
    const firstSend = Promise.withResolvers<void>();
    const onSend = vi
      .fn<(message: string) => Promise<void>>()
      .mockReturnValueOnce(firstSend.promise)
      .mockResolvedValueOnce(undefined);
    const { rerender } = render(renderOptions(onSend));

    const option = screen.getByRole<HTMLButtonElement>("button", {
      name: /^Fix enrichment/,
    });
    fireEvent.click(option);
    expect(option.disabled).toBe(true);
    fireEvent.click(option);
    expect(onSend).toHaveBeenCalledExactlyOnceWith("Fix enrichment");

    await act(async () => {
      firstSend.reject(error);
    });

    expect(
      await screen.findByText("Couldn't confirm the message was sent"),
    ).toBeDefined();
    expect(option.disabled).toBe(true);
    fireEvent.click(option);
    expect(onSend).toHaveBeenCalledTimes(1);
    expect(
      screen.getByText("Check the chat to see whether the message arrived."),
    ).toBeDefined();
    expect(screen.queryByText(/try again/i)).toBeNull();

    rerender(renderOptions(onSend, true));
    const [previousOption, laterOption] =
      screen.getAllByRole<HTMLButtonElement>("button", {
        name: /^Fix enrichment/,
      });
    expect(previousOption.disabled).toBe(true);
    expect(laterOption.disabled).toBe(false);
    await act(async () => {
      fireEvent.click(laterOption);
    });
    expect(onSend).toHaveBeenNthCalledWith(2, "Fix enrichment");
    expect(laterOption.disabled).toBe(true);
  },
);

it.each([
  [new CopilotAuthError(), "Couldn't send the message"],
  [new SendInFlightError(), "Couldn't send the message"],
  [new QueueFollowUpFullError(), "Couldn't send the message"],
  [new ApiError("No plan", 402, null), "Sending needs an active plan"],
])(
  "lets the user retry after a definite send rejection (%s)",
  async (error, title) => {
    const onSend = vi
      .fn<(message: string) => Promise<void>>()
      .mockRejectedValueOnce(error)
      .mockResolvedValueOnce(undefined);
    render(renderOptions(onSend));

    const option = screen.getByRole<HTMLButtonElement>("button", {
      name: /^Fix enrichment/,
    });
    await act(async () => {
      fireEvent.click(option);
    });
    expect(await screen.findByText(title)).toBeDefined();
    expect(option.disabled).toBe(false);

    await act(async () => {
      fireEvent.click(option);
    });
    expect(onSend).toHaveBeenNthCalledWith(2, "Fix enrichment");
    expect(option.disabled).toBe(true);
    fireEvent.click(option);
    expect(onSend).toHaveBeenCalledTimes(2);
  },
);
