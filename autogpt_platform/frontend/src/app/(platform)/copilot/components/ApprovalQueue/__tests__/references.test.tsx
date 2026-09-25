import userEvent from "@testing-library/user-event";
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

test("hovering a resolved link shows its summary and its ID", async () => {
  renderCard(referenceCard("Move agents"));

  const view = await card();
  await userEvent.hover(view.getByRole("link", { name: "Morning digest" }));

  const tip = await screen.findByRole("tooltip");
  expect(tip.textContent).toContain(
    "Summarises overnight email and news at 7am.",
  );
  expect(tip.textContent).toContain("lib-digest");
});

test("a link with no summary has no tooltip, only the ID as its title", async () => {
  const review = referenceCard("Move agents");
  const payload = review.payload as { references: { summary: unknown }[] };
  payload.references.forEach((ref) => (ref.summary = null));
  renderCard(review);

  const view = await card();
  const digest = view.getByRole("link", { name: "Morning digest" });
  await userEvent.hover(digest);

  expect(digest.getAttribute("title")).toBe("lib-digest");
  await new Promise((resolve) => setTimeout(resolve, 400));
  expect(screen.queryByRole("tooltip")).toBeNull();
});

test("a list the server clipped still counts every ID it held", async () => {
  const review = referenceCard("Move agents");
  const payload = review.payload as {
    arguments: Record<string, unknown>;
    clipped: string[];
    reference_totals: Record<string, number>;
  };
  payload.arguments.agent_ids = '["lib-digest", "lib-gone", "lib-tri…';
  payload.clipped = ["agent_ids"];
  payload.reference_totals.agent_ids = 120;
  renderCard(review);

  const view = await card();
  expect(view.getByRole("link", { name: "Morning digest" })).toBeDefined();
  expect(view.getByText(/\+115 more/)).toBeDefined();
});

test("unresolved agent IDs stay listed beside a resolved folder headline", async () => {
  const review = referenceCard("Move agents");
  const payload = review.payload as {
    arguments: Record<string, unknown>;
    references: { key: string; name: string | null; href: string | null }[];
    reference_totals: Record<string, number>;
  };
  payload.arguments.agent_ids = ["lib-lost-1", "lib-lost-2"];
  payload.references = [
    payload.references.find((ref) => ref.key === "folder_id")!,
    ...["lib-lost-1", "lib-lost-2"].map((id) => ({
      key: "agent_ids",
      entity: "library_agent",
      id,
      name: null,
      href: null,
      summary: null,
    })),
  ];
  payload.reference_totals = { folder_id: 1, agent_ids: 2 };
  renderCard(review);

  const view = await card();
  expect(
    view.getByRole("heading", { name: /Move agents into a folder Archive/ }),
  ).toBeDefined();
  expect(view.getByText(/lib-lost-1/)).toBeDefined();
  expect(view.getByText(/lib-lost-2/)).toBeDefined();
});
