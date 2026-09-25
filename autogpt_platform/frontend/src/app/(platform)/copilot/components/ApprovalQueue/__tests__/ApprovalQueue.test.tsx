import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { expect, test, vi } from "vitest";
import { getPostV2ProcessReviewActionMockHandler200 } from "@/app/api/__generated__/endpoints/executions/executions.msw";
import type { PendingHumanReviewModel } from "@/app/api/__generated__/models/pendingHumanReviewModel";
import { server } from "@/mocks/mock-server";
import {
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import { CopilotChatActionsProvider } from "../../CopilotChatActionsProvider/CopilotChatActionsProvider";
import { CopilotPendingReviews } from "../../CopilotPendingReviews/CopilotPendingReviews";
import {
  deleteFolder,
  folder,
  heldReview,
  mail,
  CHAT_SESSION,
  shell,
} from "./fixtures";

function serve(reviews: PendingHumanReviewModel[], status = 200) {
  server.use(
    http.get(`*/api/review/session/${CHAT_SESSION}`, () =>
      HttpResponse.json(reviews),
    ),
    status === 200
      ? getPostV2ProcessReviewActionMockHandler200({
          approved_count: 1,
          rejected_count: 0,
          failed_count: 0,
        })
      : http.post("*/api/review/action", () =>
          HttpResponse.json({ detail: "boom" }, { status }),
        ),
  );
}

function renderQueue() {
  return render(
    <CopilotChatActionsProvider onSend={vi.fn()} onBackendTurn={vi.fn()}>
      <CopilotPendingReviews chatSessionId={CHAT_SESSION} />
    </CopilotChatActionsProvider>,
  );
}

async function queue() {
  return screen.findByRole("region", { name: "Waiting for you" });
}

test("the headline names the action and its argument is not repeated below", async () => {
  serve([
    heldReview({
      id: "a",
      tool: "create_folder",
      args: { name: "Q3 reports", color: "blue" },
      fields: [
        { key: "name", label: "Name" },
        { key: "color", label: "Color" },
      ],
      headline: {
        ask: "Create folder",
        object: "Q3 reports",
        object_key: "name",
      },
    }),
  ]);
  renderQueue();

  const heading = await screen.findByRole("heading", {
    name: "Create folder Q3 reports",
  });
  expect(heading).toBeDefined();
  expect(screen.getByText("Color")).toBeDefined();
  expect(screen.queryByText("Name")).toBeNull();
  expect(screen.getAllByText("Q3 reports")).toHaveLength(1);
});

test("a secret renders hidden, never its value", async () => {
  serve([mail()]);
  renderQueue();

  await queue();
  expect(screen.getByText("Api key")).toBeDefined();
  expect(screen.getByText("hidden")).toBeDefined();
  expect(screen.queryByText("[redacted]")).toBeNull();
});

test("the mode's reason is said once, in the header, and on no card", async () => {
  serve([folder("a", "Q3 reports"), folder("b", "Invoices")]);
  renderQueue();

  await queue();
  expect(
    screen.getAllByText(
      /Ask First is on, so Otto asks before he changes anything outside his workspace/,
    ),
  ).toHaveLength(1);
  expect(screen.queryByText(/so this action needs your approval/)).toBeNull();
});

test("approve-all is offered for two cards on the same subject", async () => {
  serve([folder("a", "Q3 reports"), folder("b", "Invoices")]);
  renderQueue();

  expect(
    await screen.findByRole("button", { name: "Approve both" }),
  ).toBeDefined();
});

test.each([
  ["a mixed pair", () => [folder("a", "Q3 reports"), shell()]],
  ["an irreversible pair", () => [mail("m1", []), mail("m2", [])]],
])("approve-all is not offered for %s", async (_name, reviews) => {
  serve(reviews());
  renderQueue();

  await queue();
  expect(
    screen.queryByRole("button", { name: /Approve (both|all)/ }),
  ).toBeNull();
});

test("from four cards up, a line with fields offers Review, not Approve", async () => {
  serve([
    folder("a", "One", 9),
    folder("b", "Two", 8),
    folder("c", "Three", 7),
    shell(),
  ]);
  renderQueue();

  await queue();
  const lines = screen.getAllByRole("listitem");
  const shellLine = lines.find((li) => within(li).queryByText(/sandbox/));
  expect(shellLine).toBeDefined();
  expect(
    within(shellLine!).getByRole("button", { name: "Review" }),
  ).toBeDefined();
  expect(
    within(shellLine!).queryByRole("button", { name: "Approve" }),
  ).toBeNull();
  expect(screen.getAllByRole("button", { name: "Approve" })).toHaveLength(3);
});

test("the approve menu is absent when the server offers no chat rule", async () => {
  serve([mail("m1", [])]);
  renderQueue();

  await queue();
  expect(screen.getByRole("button", { name: "Approve" })).toBeDefined();
  expect(
    screen.queryByRole("button", { name: "More ways to approve" }),
  ).toBeNull();
});

test("the approve menu offers the rules the server allows", async () => {
  serve([mail("m1", ["allow"])]);
  renderQueue();

  await userEvent.click(
    await screen.findByRole("button", { name: "More ways to approve" }),
  );
  expect(await screen.findByText("Approve for this chat")).toBeDefined();
  expect(screen.queryByText(/judge from now on/)).toBeNull();
  expect(screen.queryByText(/Always allow/)).toBeNull();
});

test("a failed send says so on the card and lets you try again", async () => {
  serve([folder("a", "Q3 reports")], 500);
  renderQueue();

  await userEvent.click(await screen.findByRole("button", { name: "Approve" }));

  expect((await screen.findByRole("alert")).textContent).toContain(
    "Couldn't send your answer. Nothing ran. Try again.",
  );
  await waitFor(() =>
    expect(
      (screen.getByRole("button", { name: "Approve" }) as HTMLButtonElement)
        .disabled,
    ).toBe(false),
  );
});

test.each([
  ["supervisor", "Not sure this is safe: it uploads a file."],
  ["rule", "it uploads a file."],
])("a %s reason about this call shows on its card", async (kind, line) => {
  serve([
    heldReview({
      id: "r",
      tool: "bash_exec",
      args: { command: "ls" },
      reason: "it uploads a file.",
      reasonKind: kind,
      headline: { ask: "Run a command in the sandbox" },
    }),
  ]);
  renderQueue();

  expect(await screen.findByText(line)).toBeDefined();
});

test("a call told apart only by its id shows the id and is not approved as a set", async () => {
  serve([deleteFolder("a", "f-111"), deleteFolder("b", "f-222")]);
  renderQueue();

  expect(await screen.findByText("f-111")).toBeDefined();
  expect(screen.getByText("f-222")).toBeDefined();
  expect(screen.queryByRole("button", { name: "Approve both" })).toBeNull();
  // A folder delete moves its agents to the root, so it is not irreversible.
  expect(screen.queryByText("Can't be undone")).toBeNull();
});

test("a failed answer from a compact line opens its card with the error", async () => {
  serve(
    [
      folder("a", "One", 9),
      folder("b", "Two", 8),
      folder("c", "Three", 7),
      folder("d", "Four", 6),
    ],
    500,
  );
  renderQueue();

  const [first] = await screen.findAllByRole("button", { name: "Approve" });
  await userEvent.click(first);

  expect((await screen.findByRole("alert")).textContent).toContain(
    "Couldn't send your answer",
  );
});

// As the server marks them (backend/copilot/gate/policy.py `is_irreversible`).
function destructive(id: string, tool: string, args: Record<string, unknown>) {
  return heldReview({
    id,
    tool,
    args,
    subject: {
      kind: "tool",
      key: tool,
      name: tool,
      effect: "platform",
      irreversible: true,
    },
  });
}

test.each([
  ["delete_workspace_file", (n: string) => ({ path: `${n}.md` })],
  ["delete_skill", (n: string) => ({ name: n })],
  ["delete_schedule", (n: string) => ({ schedule_id: n })],
  ["memory_forget_confirm", (n: string) => ({ uuids: [n], hard_delete: true })],
])(
  "%s is marked can't-be-undone and never approved as a set",
  async (tool, args) => {
    serve([
      destructive("a", tool, args("one")),
      destructive("b", tool, args("two")),
    ]);
    renderQueue();

    await queue();
    expect(screen.getAllByText("Can't be undone")).toHaveLength(2);
    expect(screen.queryByRole("button", { name: "Approve both" })).toBeNull();
  },
);
