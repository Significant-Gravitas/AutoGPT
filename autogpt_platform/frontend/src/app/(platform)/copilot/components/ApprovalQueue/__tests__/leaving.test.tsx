import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { afterEach, beforeEach, expect, test, vi } from "vitest";
import type { PendingHumanReviewModel } from "@/app/api/__generated__/models/pendingHumanReviewModel";
import { server } from "@/mocks/mock-server";
import { act, render, screen, within } from "@/tests/integrations/test-utils";
import { CopilotChatActionsProvider } from "../../CopilotChatActionsProvider/CopilotChatActionsProvider";
import { CopilotPendingReviews } from "../../CopilotPendingReviews/CopilotPendingReviews";
import { useHeldAnswersStore } from "../heldAnswersStore";
import { CHAT_SESSION, folder } from "./fixtures";

// The server's list: a review leaves it once it is answered.
let waiting: PendingHumanReviewModel[] = [];

beforeEach(() => {
  vi.useFakeTimers({ shouldAdvanceTime: true });
  server.use(
    http.get(`*/api/review/session/${CHAT_SESSION}`, () =>
      HttpResponse.json(waiting),
    ),
    http.post("*/api/review/action", async ({ request }) => {
      const { reviews } = (await request.json()) as {
        reviews: { node_exec_id: string }[];
      };
      const answered = new Set(reviews.map((r) => r.node_exec_id));
      waiting = waiting.filter((r) => !answered.has(r.node_exec_id));
      return HttpResponse.json({
        approved_count: reviews.length,
        rejected_count: 0,
        failed_count: 0,
      });
    }),
  );
});

afterEach(() => {
  vi.useRealTimers();
  useHeldAnswersStore.setState({ answers: {} });
});

test("the box leaves about a second after its last card is answered", async () => {
  waiting = [folder("a", "Q3 reports")];
  const user = renderQueue();

  await approve(user, "Create library folder Q3 reports");
  const answeredAt = Date.now();
  for (let i = 0; i < 30 && box(); i++) await advance(100);

  expect(box()).toBeNull();
  expect(Date.now() - answeredAt).toBeGreaterThanOrEqual(900);
});

test("the box and its receipts stay while a card still waits", async () => {
  waiting = [folder("a", "Q3 reports"), folder("b", "Invoices")];
  const user = renderQueue();

  await approve(user, "Create library folder Q3 reports");
  await advance(5000);

  expect(box()).not.toBeNull();
  expect(screen.getByText("· Approved")).toBeDefined();
  expect(
    screen.getByRole("heading", { name: "Create library folder Invoices" }),
  ).toBeDefined();
});

test("a new held call brings the box back without the earlier receipt", async () => {
  waiting = [folder("a", "Q3 reports")];
  const user = renderQueue();

  await approve(user, "Create library folder Q3 reports");
  await advance(1500);
  expect(box()).toBeNull();

  waiting = [folder("b", "Invoices")];
  await advance(2500);

  expect(
    await screen.findByRole("heading", {
      name: "Create library folder Invoices",
    }),
  ).toBeDefined();
  expect(screen.queryByText("· Approved")).toBeNull();
});

function renderQueue() {
  render(
    <CopilotChatActionsProvider onSend={vi.fn()} onBackendTurn={vi.fn()}>
      <CopilotPendingReviews chatSessionId={CHAT_SESSION} />
    </CopilotChatActionsProvider>,
  );
  return userEvent.setup({ advanceTimers: vi.advanceTimersByTime });
}

function box() {
  return screen.queryByRole("region", { name: "Waiting for you" });
}

async function approve(user: ReturnType<typeof userEvent.setup>, name: string) {
  const card = (await screen.findByRole("heading", { name })).closest("li")!;
  await user.click(within(card).getByRole("button", { name: "Approve" }));
  await screen.findByText("· Approved");
}

async function advance(ms: number) {
  await act(() => vi.advanceTimersByTimeAsync(ms));
}
