import { http, HttpResponse } from "msw";
import { expect, test, vi } from "vitest";
import type { PendingHumanReviewModel } from "@/app/api/__generated__/models/pendingHumanReviewModel";
import { server } from "@/mocks/mock-server";
import { render, screen } from "@/tests/integrations/test-utils";
import { CopilotChatActionsProvider } from "../../CopilotChatActionsProvider/CopilotChatActionsProvider";
import { CopilotPendingReviews } from "../../CopilotPendingReviews/CopilotPendingReviews";
import { CHAT_SESSION, spendCard } from "./fixtures";

function renderQueue(reviews: PendingHumanReviewModel[]) {
  server.use(
    http.get(`*/api/review/session/${CHAT_SESSION}`, () =>
      HttpResponse.json(reviews),
    ),
  );
  render(
    <CopilotChatActionsProvider onSend={vi.fn()} onBackendTurn={vi.fn()}>
      <CopilotPendingReviews chatSessionId={CHAT_SESSION} />
    </CopilotChatActionsProvider>,
  );
}

test("a spend card shows this step's cost, the task's spend against its ceiling, and what approving adds", async () => {
  renderQueue([spendCard()]);

  expect(
    await screen.findByRole("heading", { name: /Run Perplexity Search/ }),
  ).toBeDefined();
  expect(screen.getByText("about $0.05")).toBeDefined();
  expect(screen.getByText("$2.41 of $2.00")).toBeDefined();
  expect(screen.getByText("Approving runs this step.")).toBeDefined();
  expect(
    screen.getByRole("progressbar", { name: "Spent of this turn's budget" }),
  ).toBeDefined();
  // The money block says it; the model's sentence is not repeated on the card.
  expect(screen.queryByText(/costs about/)).toBeNull();
});

test("a spend card offers no rule even when one is listed, and is never approved as a set", async () => {
  renderQueue([spendCard("a", ["allow", "judge"]), spendCard("b")]);

  await screen.findAllByRole("heading", { name: /Run Perplexity Search/ });
  expect(
    screen.queryByRole("button", { name: "More ways to approve" }),
  ).toBeNull();
  expect(screen.queryByRole("button", { name: "Approve both" })).toBeNull();
});
