import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { expect, test, vi } from "vitest";
import { server } from "@/mocks/mock-server";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import { CopilotChatActionsProvider } from "../../CopilotChatActionsProvider/CopilotChatActionsProvider";
import { CopilotPendingReviews } from "../../CopilotPendingReviews/CopilotPendingReviews";
import { mail, mcpTool, CHAT_SESSION } from "./fixtures";

function serve(
  review: ReturnType<typeof mail> = mail("m1", ["allow", "judge"]),
) {
  const sent: { reviews: Record<string, unknown>[] }[] = [];
  server.use(
    http.get(`*/api/review/session/${CHAT_SESSION}`, () =>
      HttpResponse.json([review]),
    ),
    http.post("*/api/review/action", async ({ request }) => {
      sent.push(
        (await request.json()) as { reviews: Record<string, unknown>[] },
      );
      return HttpResponse.json({
        approved_count: 1,
        rejected_count: 0,
        failed_count: 0,
      });
    }),
  );
  return sent;
}

function renderQueue() {
  render(
    <CopilotChatActionsProvider onSend={vi.fn()} onBackendTurn={vi.fn()}>
      <CopilotPendingReviews chatSessionId={CHAT_SESSION} />
    </CopilotChatActionsProvider>,
  );
}

test.each([
  ["Approve for this chat", "allow", "a block", mail("m1")],
  ["Let Otto judge from now on", "judge", "a block", mail("m1")],
  ["Approve for this chat", "allow", "an MCP tool", mcpTool("m1")],
  ["Let Otto judge from now on", "judge", "an MCP tool", mcpTool("m1")],
])(
  "choosing %s approves the call and sets the %s rule on %s",
  async (item, rule, _subject, review) => {
    const sent = serve(review);
    renderQueue();

    await userEvent.click(
      await screen.findByRole("button", { name: "More ways to approve" }),
    );
    await userEvent.click(await screen.findByText(item));

    await waitFor(() => expect(sent).toHaveLength(1));
    expect(sent[0].reviews).toEqual([
      expect.objectContaining({ approved: true, chat_rule: rule }),
    ]);
  },
);

test("a plain approve sets no rule", async () => {
  const sent = serve();
  renderQueue();

  await userEvent.click(await screen.findByRole("button", { name: "Approve" }));

  await waitFor(() => expect(sent).toHaveLength(1));
  expect(sent[0].reviews).toEqual([
    expect.objectContaining({ approved: true, chat_rule: null }),
  ]);
});
