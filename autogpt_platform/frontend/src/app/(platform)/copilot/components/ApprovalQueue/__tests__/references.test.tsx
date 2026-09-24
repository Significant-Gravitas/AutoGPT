import { http, HttpResponse } from "msw";
import { expect, test, vi } from "vitest";
import type { PendingHumanReviewModel } from "@/app/api/__generated__/models/pendingHumanReviewModel";
import { server } from "@/mocks/mock-server";
import { render, screen, within } from "@/tests/integrations/test-utils";
import { CopilotChatActionsProvider } from "../../CopilotChatActionsProvider/CopilotChatActionsProvider";
import { CopilotPendingReviews } from "../../CopilotPendingReviews/CopilotPendingReviews";
import { CHAT_SESSION, referenceCard } from "./fixtures";

function renderCard(review: PendingHumanReviewModel) {
  server.use(
    http.get(`*/api/review/session/${CHAT_SESSION}`, () =>
      HttpResponse.json([review]),
    ),
  );
  render(
    <CopilotChatActionsProvider onSend={vi.fn()} onBackendTurn={vi.fn()}>
      <CopilotPendingReviews chatSessionId={CHAT_SESSION} />
    </CopilotChatActionsProvider>,
  );
}

async function card() {
  return within(await screen.findByRole("article"));
}

test("a held delete names its folder in the headline, not as a raw id", async () => {
  renderCard(referenceCard("Delete folder"));

  const view = await card();
  expect(
    view.getByRole("heading", { name: /Delete a folder Q3 reports/ }),
  ).toBeDefined();
  expect(view.queryByText("f-q3")).toBeNull();
  const folder = view.getByRole("link", { name: "Q3 reports" });
  expect(folder.getAttribute("href")).toBe("/library?folder=f-q3");
});

test("a list of ids reads as linked names, the unresolved one as its id", async () => {
  renderCard(referenceCard("Move agents"));

  const view = await card();
  const digest = view.getByRole("link", { name: "Morning digest" });
  expect(digest.getAttribute("href")).toBe("/library/agents/lib-digest");
  expect(digest.getAttribute("title")).toBe("lib-digest");
  expect(view.getByText("lib-gone")).toBeDefined();
  expect(view.queryByRole("link", { name: "lib-gone" })).toBeNull();
  expect(view.getByText(/\+2 more/)).toBeDefined();
});

test("a resolved id links to its page", async () => {
  renderCard(referenceCard("Hire expert"));

  const view = await card();
  const template = view.getByRole("link", { name: "Ada" });
  expect(template.getAttribute("href")).toBe("/marketplace/experts/tpl-ada");
});

test("an id nothing resolved stays the raw id with no link", async () => {
  renderCard(referenceCard("Unresolved id"));

  const view = await card();
  expect(view.getByRole("heading", { name: "Delete a preset" })).toBeDefined();
  expect(view.getByText("3f0c9a2e-preset-gone")).toBeDefined();
  expect(view.queryAllByRole("link")).toHaveLength(0);
});

test("a stored link that leaves the app is dropped, the name kept", async () => {
  const review = referenceCard("Hire expert");
  const payload = review.payload as { references: { href: string }[] };
  payload.references[0].href = "https://evil.example/phish";
  renderCard(review);

  const view = await card();
  expect(view.getByText("Ada", { selector: "span" })).toBeDefined();
  expect(view.queryAllByRole("link")).toHaveLength(0);
});
