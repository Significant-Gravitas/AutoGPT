import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { expect, test, vi } from "vitest";
import { server } from "@/mocks/mock-server";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import type { MessagePart } from "../../ChatMessagesContainer/helpers";
import { CopilotChatActionsProvider } from "../../CopilotChatActionsProvider/CopilotChatActionsProvider";
import { CopilotPendingReviews } from "../../CopilotPendingReviews/CopilotPendingReviews";
import { ToolChain } from "../../ToolChain/ToolChain";
import { approvalCardId } from "../helpers";
import { folder, CHAT_SESSION } from "./fixtures";

// The chain row sits above the queue; its link must reach a card that is in
// the page however far the queue is scrolled, compact lines included.
test.each([1, 4])(
  "Go to the approval focuses the held call's card with %i waiting",
  async (count) => {
    const reviews = Array.from({ length: count }, (_, i) =>
      folder(`id${i}`, `Folder ${i}`, 10 - i),
    );
    server.use(
      http.get(`*/api/review/session/${CHAT_SESSION}`, () =>
        HttpResponse.json(reviews),
      ),
    );
    const held = {
      type: "tool-create_folder",
      state: "output-available",
      toolCallId: "call-id0",
      input: { name: "Folder 0" },
      output: {
        type: "approval_required",
        tool_name: "create_folder",
        review_id: reviews[0].node_exec_id,
        ask: "Create folder",
        object: "Folder 0",
      },
    } as MessagePart;

    render(
      <CopilotChatActionsProvider onSend={vi.fn()}>
        <ToolChain parts={[held]} isStreaming={false} />
        <CopilotPendingReviews chatSessionId={CHAT_SESSION} />
      </CopilotChatActionsProvider>,
    );

    await screen.findByRole("region", { name: "Waiting for you" });
    await userEvent.click(
      await screen.findByRole("button", { name: "Go to the approval" }),
    );

    await waitFor(() =>
      expect(document.activeElement?.id).toBe(
        approvalCardId(reviews[0].node_exec_id),
      ),
    );
  },
);
