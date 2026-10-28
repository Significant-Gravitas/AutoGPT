import { http, HttpResponse } from "msw";
import { expect, test, vi } from "vitest";
import { server } from "@/mocks/mock-server";
import { render, screen, within } from "@/tests/integrations/test-utils";
import { CopilotChatActionsProvider } from "../../CopilotChatActionsProvider/CopilotChatActionsProvider";
import { CopilotPendingReviews } from "../../CopilotPendingReviews/CopilotPendingReviews";
import { CHAT_SESSION, realCardSchemaHandler, realCards } from "./fixtures";

const cards = realCards();

function renderCard(story: string) {
  const card = cards.find((c) => c.story === story)!;
  server.use(
    http.get(`*/api/review/session/${CHAT_SESSION}`, () =>
      HttpResponse.json([card.review]),
    ),
    realCardSchemaHandler(cards),
  );
  render(
    <CopilotChatActionsProvider onSend={vi.fn()} onBackendTurn={vi.fn()}>
      <CopilotPendingReviews chatSessionId={CHAT_SESSION} />
    </CopilotChatActionsProvider>,
  );
}

test("a new Airtable table is shown by name and fields before it is made", async () => {
  renderCard("Airtable Create Table");

  expect(
    await screen.findByRole("heading", {
      name: "Add a table to your Airtable: Printify orders",
    }),
  ).toBeDefined();
  const preview = screen.getByRole("list", {
    name: "Fields in Printify orders",
  });
  const rows = within(preview)
    .getAllByRole("listitem")
    .map((row) => row.textContent);
  expect(rows).toEqual([
    "Order numberText",
    "CustomerText",
    "Totalprecision 2Money",
    "StatusChoice of New, Shipped",
    "Ordered onDate and time",
  ]);
  expect(screen.queryByText("View as JSON")).toBeNull();
  expect(screen.queryByText("Table fields")).toBeNull();
  // Names go to Airtable as written, so a browser translation must not touch them.
  for (const name of ["Printify orders", "Order number", "Ordered on"]) {
    const shown = within(
      screen.getByRole("region", { name: "Waiting for you" }),
    )
      .getAllByText(name)
      .filter((el) => el.closest("ul, header"));
    expect(shown.length).toBeGreaterThan(0);
    for (const el of shown) expect(el.getAttribute("translate")).toBe("no");
  }
});

test("a card with nothing to preview still says which step it runs", async () => {
  renderCard("Gmail Send");

  expect(
    await screen.findByRole("heading", { name: /Run Gmail Send/ }),
  ).toBeDefined();
  expect(screen.queryByRole("list", { name: /Fields in/ })).toBeNull();
});
