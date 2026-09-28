import {
  getGetV2GetPendingReviewsForChatSessionMockHandler200,
  getPostV2ProcessReviewActionMockHandler200,
} from "@/app/api/__generated__/endpoints/executions/executions.msw";
import { server } from "@/mocks/mock-server";
import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import { cleanup } from "@testing-library/react";
import { http, HttpResponse } from "msw";
import { afterEach, describe, expect, it, vi } from "vitest";
import { heldReview } from "../../ApprovalQueue/__tests__/fixtures";
import { CopilotChatActionsProvider } from "../../CopilotChatActionsProvider/CopilotChatActionsProvider";
import { WorkTab } from "../components/WorkTab/WorkTab";

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return { ...actual, useGetFlag: () => false };
});

const { toast } = vi.hoisted(() => ({ toast: vi.fn() }));
vi.mock("@/components/molecules/Toast/use-toast", () => ({
  toast,
  useToast: () => ({ toast, toasts: [], dismiss: vi.fn() }),
}));

const SESSION_ROUTE = "/api/proxy/api/chat/sessions/:sessionId";
const ALEX = { id: "exp-alex", name: "Alex", role: "PM", avatar_url: null };

/** The chat holds one hand-off with `output`; the teammate's session is
 *  `sub` (idle unless said otherwise). */
function chatWith(
  output: Record<string, unknown>,
  sub: Record<string, unknown> = {},
) {
  server.use(
    http.get(SESSION_ROUTE, ({ params }) => {
      const id = String(params.sessionId);
      const isChat = id === "chat-1";
      return HttpResponse.json({
        id,
        created_at: "2026-09-28T10:00:00Z",
        updated_at: "2026-09-28T10:00:00Z",
        user_id: "u-1",
        chat_status: "idle",
        messages: isChat
          ? [
              {
                role: "assistant",
                content: "",
                tool_calls: [
                  {
                    id: "call-1",
                    function: {
                      name: "delegate_to_expert",
                      arguments: JSON.stringify({
                        expert_id: "exp-alex",
                        prompt: "Draft the PRD",
                      }),
                    },
                  },
                ],
              },
              {
                role: "tool",
                tool_call_id: "call-1",
                content: JSON.stringify(output),
              },
            ]
          : [],
        ...(isChat ? {} : sub),
      });
    }),
  );
}

async function openDetail(onSend = vi.fn()) {
  render(
    <CopilotChatActionsProvider onSend={onSend}>
      <WorkTab sessionId="chat-1" />
    </CopilotChatActionsProvider>,
  );
  fireEvent.click(await screen.findByTestId("delegation-row"));
  return screen.findByTestId("delegation-detail");
}

describe("Work panel detail", () => {
  afterEach(cleanup);

  it("says a queued teammate starts when free and can be cancelled", async () => {
    chatWith(
      { status: "queued", sub_session_id: "sub-1", expert: ALEX },
      { chat_status: "queued" },
    );
    await openDetail();
    expect(
      await screen.findByText("Alex starts as soon as they are free."),
    ).toBeDefined();
    expect(screen.getByRole("button", { name: "Cancel" })).toBeDefined();
  });

  it("says so when a teammate could not be stopped", async () => {
    chatWith(
      { status: "running", sub_session_id: "sub-1", expert: ALEX },
      { chat_status: "running" },
    );
    server.use(
      http.post(
        `${SESSION_ROUTE}/cancel`,
        () => new HttpResponse(null, { status: 500 }),
      ),
    );
    await openDetail();
    fireEvent.click(await screen.findByRole("button", { name: /cancel/i }));
    await waitFor(() =>
      expect(toast).toHaveBeenCalledWith(
        expect.objectContaining({ title: "Couldn't stop Alex" }),
      ),
    );
  });

  it("nudges a working teammate through Otto", async () => {
    const onSend = vi.fn();
    chatWith(
      { status: "running", sub_session_id: "sub-1", expert: ALEX },
      { chat_status: "running" },
    );
    await openDetail(onSend);
    fireEvent.click(await screen.findByRole("button", { name: /nudge/i }));
    expect(onSend).toHaveBeenCalledWith("Please check on Alex's hand-off.");
  });

  it("shows what came back, files and all, and re-delegates", async () => {
    const onSend = vi.fn();
    chatWith({
      status: "completed",
      sub_session_id: "sub-1",
      response: "PRD drafted.",
      expert: ALEX,
      started_at: "2026-09-28T10:41:00Z",
      finished_at: "2026-09-28T10:48:00Z",
      elapsed_seconds: 400,
      cost_usd: 0.31,
      sub_workspace_files: [
        { name: "prd.md", path: "/p/prd.md", size_bytes: 12_288 },
        { name: "big.pdf", path: "/p/big.pdf", size_bytes: 3_145_728 },
        { name: "tiny.txt", path: "/p/tiny.txt", size_bytes: 12 },
      ],
    });
    const detail = await openDetail(onSend);
    expect(await screen.findByText("What came back")).toBeDefined();
    expect(detail.textContent).toContain("PRD drafted.");
    expect(screen.getByText("12 KB")).toBeDefined();
    expect(screen.getByText("3.0 MB")).toBeDefined();
    expect(screen.getByText("12 B")).toBeDefined();
    expect(detail.textContent).toContain("6m 40s · $0.31");
    fireEvent.click(screen.getByRole("button", { name: /re-delegate/i }));
    expect(onSend).toHaveBeenCalledWith(
      "Please re-delegate the same brief to Alex.",
    );
  });

  it("answers the teammate's own held call from Otto's chat", async () => {
    chatWith(
      { status: "running", sub_session_id: "sub-1", expert: ALEX },
      { chat_status: "idle" },
    );
    let processed = false;
    server.use(
      getGetV2GetPendingReviewsForChatSessionMockHandler200([
        {
          ...heldReview({
            id: "r1",
            tool: "send_email",
            headline: { ask: "Send an email" },
          }),
          session_id: "sub-1",
        },
      ]),
      http.post("*/api/review/action", () => {
        processed = true;
        return HttpResponse.json({
          approved_count: 1,
          rejected_count: 0,
          failed_count: 0,
          error: null,
        });
      }),
    );
    await openDetail();
    const reviews = await screen.findByTestId("sub-session-reviews");
    expect(reviews.textContent).toContain("Send an email");
    fireEvent.click(screen.getByRole("button", { name: /^approve$/i }));
    await waitFor(() => expect(processed).toBe(true));
  });

  it("flags a failed answer to the teammate's held call", async () => {
    chatWith({ status: "running", sub_session_id: "sub-1", expert: ALEX });
    server.use(
      getGetV2GetPendingReviewsForChatSessionMockHandler200([
        {
          ...heldReview({ id: "r1", tool: "send_email" }),
          session_id: "sub-1",
        },
      ]),
      getPostV2ProcessReviewActionMockHandler200({
        approved_count: 0,
        rejected_count: 0,
        failed_count: 1,
        error: "nope",
      }),
    );
    await openDetail();
    await screen.findByTestId("sub-session-reviews");
    fireEvent.click(screen.getByRole("button", { name: /reject/i }));
    expect(await screen.findByRole("alert")).toBeDefined();
  });
});
