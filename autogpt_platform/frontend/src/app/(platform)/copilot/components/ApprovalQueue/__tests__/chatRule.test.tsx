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
  ["Let Otto judge Gmail Send", "judge", "a block", mail("m1")],
  ["Approve for this chat", "allow", "an MCP tool", mcpTool("m1")],
  [
    "Let Otto judge create_issue on mcp.linear.app",
    "judge",
    "an MCP tool",
    mcpTool("m1"),
  ],
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

const TEAM = "Apply to all Experts in my Team";

test("the team toggle starts off and a rule then holds for this chat only", async () => {
  const sent = serve();
  renderQueue();

  await userEvent.click(
    await screen.findByRole("button", { name: "More ways to approve" }),
  );
  const toggle = await screen.findByRole("menuitemcheckbox", { name: TEAM });
  expect(toggle.getAttribute("aria-checked")).toBe("false");
  await userEvent.click(screen.getByText("Approve for this chat"));

  await waitFor(() => expect(sent).toHaveLength(1));
  expect(sent[0].reviews[0]).not.toHaveProperty("apply_to_team");
});

const SUBJECT = "Gmail Send";

test.each([
  [
    "off",
    false,
    [
      "Approve for this chat",
      `${SUBJECT} runs without asking until this chat ends`,
      `Let Otto judge ${SUBJECT}`,
      "A check decides each time it runs in this chat, and asks you only when it isn't sure",
    ],
  ],
  [
    "on",
    true,
    [
      "Approve for all my chats",
      `${SUBJECT} runs without asking in all your chats until you revoke it`,
      `Let Otto judge ${SUBJECT} in all my chats`,
      "A check decides each time it runs in any of your chats, and asks you only when it isn't sure",
    ],
  ],
])(
  "with the team toggle %s, both rules name the subject and their scope",
  async (_state, on, lines) => {
    serve();
    renderQueue();

    await userEvent.click(
      await screen.findByRole("button", { name: "More ways to approve" }),
    );
    if (on)
      await userEvent.click(
        await screen.findByRole("menuitemcheckbox", { name: TEAM }),
      );
    for (const line of lines)
      expect(await screen.findByText(line)).toBeDefined();
  },
);

test.each([
  ["Approve for all my chats", "allow"],
  [`Let Otto judge ${SUBJECT} in all my chats`, "judge"],
])(
  "with the team toggle on, %s sends the %s rule for every chat",
  async (item, rule) => {
    const sent = serve();
    renderQueue();

    await userEvent.click(
      await screen.findByRole("button", { name: "More ways to approve" }),
    );
    await userEvent.click(
      await screen.findByRole("menuitemcheckbox", { name: TEAM }),
    );
    const toggle = await screen.findByRole("menuitemcheckbox", { name: TEAM });
    expect(toggle.getAttribute("aria-checked")).toBe("true");
    await userEvent.click(screen.getByText(item));

    await waitFor(() => expect(sent).toHaveLength(1));
    expect(sent[0].reviews).toEqual([
      expect.objectContaining({
        approved: true,
        chat_rule: rule,
        apply_to_team: true,
      }),
    ]);
  },
);
