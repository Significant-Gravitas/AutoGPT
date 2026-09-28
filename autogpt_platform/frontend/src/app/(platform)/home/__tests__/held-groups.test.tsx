import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { expect, test } from "vitest";
import type { HomeAttentionItem } from "@/app/api/__generated__/models/homeAttentionItem";
import { server } from "@/mocks/mock-server";
import {
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import {
  folder,
  heldRead,
  mail,
} from "../../copilot/components/ApprovalQueue/__tests__/fixtures";
import { NeedsYou } from "../components/NeedsYou/NeedsYou";
import { ada, homeHeldItem, leo, makeDashboard } from "./heldItems";

// The route refuses a request spanning chats with 409, as review/routes.py does.
function serveAnswers(
  items: HomeAttentionItem[],
  failingSession: string | null = null,
) {
  const sessionOf = new Map(
    items.map((item) => [item.review!.node_exec_id, item.review!.session_id]),
  );
  const requests: string[][] = [];
  server.use(
    http.post("*/api/review/action", async ({ request }) => {
      const body = (await request.json()) as {
        reviews: { node_exec_id: string }[];
      };
      const ids = body.reviews.map((r) => r.node_exec_id);
      requests.push(ids);
      const sessions = new Set(ids.map((id) => sessionOf.get(id)));
      if (sessions.size > 1)
        return HttpResponse.json(
          {
            detail:
              "All reviews in a single request must belong to the same execution.",
          },
          { status: 409 },
        );
      if (sessions.has(failingSession))
        return HttpResponse.json({ detail: "down" }, { status: 500 });
      return HttpResponse.json({
        approved_count: ids.length,
        rejected_count: 0,
        failed_count: 0,
      });
    }),
  );
  return requests;
}

function renderTile(items: HomeAttentionItem[]) {
  const view = render(<NeedsYou dashboard={makeDashboard(items)} />);
  return {
    refetch: (next: HomeAttentionItem[]) =>
      view.rerender(<NeedsYou dashboard={makeDashboard(next)} />),
  };
}

function adaFolders(count: number) {
  return Array.from({ length: count }, (_, i) =>
    homeHeldItem(folder(`a${i}`, `Folder ${i}`), {
      expert: ada,
      session: `ada-${i}`,
    }),
  );
}

test("an Expert with two held calls gets a header; a lone call stays a plain row", () => {
  renderTile([
    ...adaFolders(2),
    homeHeldItem(folder("l0", "Leads"), { expert: leo, session: "leo-0" }),
  ]);

  const group = screen.getByRole("region", { name: "Ada" });
  expect(within(group).getByRole("heading", { name: "Ada" })).toBeDefined();
  expect(within(group).getByLabelText("2 waiting")).toBeDefined();
  expect(screen.queryByRole("region", { name: "Leo" })).toBeNull();
});

test("Reject all in a group sends one request per chat and leaves receipts", async () => {
  const user = userEvent.setup();
  const items = adaFolders(3);
  const requests = serveAnswers(items);
  renderTile(items);

  const group = screen.getByRole("region", { name: "Ada" });
  await user.click(
    within(group).getByRole("button", { name: "Reject all 3…" }),
  );
  await user.click(within(group).getByRole("button", { name: "Reject all" }));

  await waitFor(() =>
    expect(screen.getAllByText("· Rejected · Otto was told")).toHaveLength(3),
  );
  expect(requests).toHaveLength(3);
  expect(requests.every((ids) => ids.length === 1)).toBe(true);
});

test("the tile's Reject all covers every Expert's held calls", async () => {
  const user = userEvent.setup();
  const items = [
    ...adaFolders(2),
    homeHeldItem(folder("l0", "Leads"), { expert: leo, session: "leo-0" }),
  ];
  const requests = serveAnswers(items);
  renderTile(items);

  await user.click(screen.getByRole("button", { name: "Reject all 3…" }));
  expect(
    screen.getByText(
      "Reject all 3? Each Expert will be told none of them ran.",
    ),
  ).toBeDefined();
  await user.click(
    screen.getAllByRole("button", { name: "Reject all" }).at(-1)!,
  );

  await waitFor(() => expect(requests).toHaveLength(3));
  expect(await screen.findAllByText("· Rejected · Otto was told")).toHaveLength(
    3,
  );
});

test.each([
  ["three bare calls of one subject", () => adaFolders(3), "Approve all 3"],
  [
    "a held read mixed in",
    () => [
      ...adaFolders(2),
      homeHeldItem(heldRead("r", "a.example"), { expert: ada, session: "r" }),
    ],
    null,
  ],
  [
    "a second subject mixed in",
    () => [
      ...adaFolders(2),
      homeHeldItem(mail("m"), { expert: ada, session: "m" }),
    ],
    null,
  ],
])("Approve all with %s", (_, build, label) => {
  renderTile(build());
  const group = screen.getByRole("region", { name: "Ada" });
  expect(
    within(group).queryByRole("button", { name: /^Approve (all|both)/ })
      ?.textContent ?? null,
  ).toBe(label);
});

test("a batch whose chat fails keeps that row with its error, the rest become receipts", async () => {
  const user = userEvent.setup();
  const items = adaFolders(2);
  serveAnswers(items, "ada-1");
  renderTile(items);

  await user.click(screen.getByRole("button", { name: "Approve both" }));

  expect(await screen.findByText("· Approved · Otto is on it")).toBeDefined();
  expect(screen.getByRole("alert").textContent).toBe(
    "Couldn't send your answer. Nothing ran. Try again.",
  );
  expect(
    screen.getByRole("button", {
      name: /Approve: Create library folder “Folder 1”/,
    }),
  ).toBeDefined();
});

test("a held call arriving on the poll joins the end of its group, never above", () => {
  const [a0, a1, newest] = adaFolders(3);
  const l0 = homeHeldItem(folder("l0", "Leads"), { expert: leo, session: "l" });
  const { refetch } = renderTile([a0, l0, a1]);

  // The feed puts the newcomer first (older, high priority); the list does not.
  refetch([{ ...newest, priority: "high" }, a0, l0, a1]);

  const objects = screen
    .getAllByText(/^(Folder \d|Leads)$/)
    .map((el) => el.textContent);
  expect(objects).toEqual(["Folder 0", "Folder 1", "Folder 2", "Leads"]);
});
