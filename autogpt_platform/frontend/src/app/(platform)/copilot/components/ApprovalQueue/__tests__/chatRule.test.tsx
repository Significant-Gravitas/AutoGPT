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

function renderQueue(expertName: string | null = "Frankie") {
  render(
    <CopilotChatActionsProvider onSend={vi.fn()} onBackendTurn={vi.fn()}>
      <CopilotPendingReviews
        chatSessionId={CHAT_SESSION}
        expertName={expertName}
      />
    </CopilotChatActionsProvider>,
  );
}

async function openMenu() {
  await userEvent.click(
    await screen.findByRole("button", { name: "More ways to approve" }),
  );
}

const MAIL = "Gmail Send";
const MCP = "create_issue on mcp.linear.app";

test.each([
  [`Approve ${MAIL} from now on`, "allow", mail("m1")],
  [`Let Otto judge ${MAIL} from now on`, "judge", mail("m1")],
  [`Approve ${MCP} from now on`, "allow", mcpTool("m1")],
  [`Let Otto judge ${MCP} from now on`, "judge", mcpTool("m1")],
])(
  "choosing %s approves the call and sets the %s rule for the Expert by default",
  async (item, rule, review) => {
    const sent = serve(review);
    renderQueue();

    await openMenu();
    await userEvent.click(await screen.findByText(item));

    await waitFor(() => expect(sent).toHaveLength(1));
    expect(sent[0].reviews).toEqual([
      expect.objectContaining({
        approved: true,
        chat_rule: rule,
        chat_rule_scope: "expert",
      }),
    ]);
  },
);

test("a plain approve sets no rule", async () => {
  const sent = serve();
  renderQueue();

  await userEvent.click(await screen.findByRole("button", { name: "Approve" }));

  await waitFor(() => expect(sent).toHaveLength(1));
  expect(sent[0].reviews[0]).toEqual(
    expect.objectContaining({ approved: true, chat_rule: null }),
  );
  expect(sent[0].reviews[0]).not.toHaveProperty("chat_rule_scope");
});

test.each([
  ["Frankie", "Frankie, every chat"],
  [null, "Otto, every chat"],
])(
  "the scope picker starts on the chat's own Expert (%s)",
  async (expertName, label) => {
    serve();
    renderQueue(expertName);

    await openMenu();
    const scope = await screen.findByRole("menuitemradio", { name: label });
    expect(scope.getAttribute("aria-checked")).toBe("true");
    expect(
      screen
        .getByRole("menuitemradio", { name: "This chat" })
        .getAttribute("aria-checked"),
    ).toBe("false");
  },
);

test.each([
  [
    "This chat",
    "chat",
    "Frankie won't ask again for this in this chat",
    "A check decides each time it runs in this chat, and asks you only when it isn't sure",
  ],
  [
    "Frankie, every chat",
    "expert",
    "Frankie won't ask again for this in any chat",
    "A check decides each time Frankie runs it, in any chat, and asks you only when it isn't sure",
  ],
  [
    "Every Expert on my team",
    "team",
    "No Expert on your team will ask again for this",
    "A check decides each time any Expert on your team runs it, and asks you only when it isn't sure",
  ],
])(
  "picking %s words both actions for that scope and sends it",
  async (label, scope, allowDetail, judgeDetail) => {
    const sent = serve();
    renderQueue("Frankie");

    await openMenu();
    await userEvent.click(
      await screen.findByRole("menuitemradio", { name: label }),
    );
    expect(await screen.findByText(allowDetail)).toBeDefined();
    expect(screen.getByText(judgeDetail)).toBeDefined();
    // Otto is the supervisor in every Expert's chat.
    expect(
      screen.getByText(`Let Otto judge ${MAIL} from now on`),
    ).toBeDefined();
    expect(screen.queryByText(/Let Frankie judge/)).toBeNull();
    await userEvent.click(screen.getByText(`Approve ${MAIL} from now on`));

    await waitFor(() => expect(sent).toHaveLength(1));
    expect(sent[0].reviews).toEqual([
      expect.objectContaining({ chat_rule: "allow", chat_rule_scope: scope }),
    ]);
  },
);
