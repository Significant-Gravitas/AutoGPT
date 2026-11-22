import type { ResolvedPage } from "@/app/api/__generated__/models/resolvedPage";
import { server } from "@/mocks/mock-server";
import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { expect, test, vi } from "vitest";
import PageViewPage from "../[id]/page";

vi.mock("next/navigation", async (importOriginal) => ({
  ...(await importOriginal<typeof import("next/navigation")>()),
  useParams: () => ({ id: "page-1" }),
  usePathname: () => "/pages/page-1",
  useRouter: () => ({ push: vi.fn(), replace: vi.fn(), refresh: vi.fn() }),
  useSearchParams: () => new URLSearchParams(),
}));

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => ({
  ...(await importOriginal<
    typeof import("@/services/feature-flags/use-get-flag")
  >()),
  useGetFlag: (flag: string) => flag === "pages" || flag === "collections",
  useFlagStatus: (flag: string) => ({
    enabled: flag === "pages" || flag === "collections",
    ready: true,
    answered: true,
  }),
}));

const IDEAS = { collection_id: "ideas", collection_name: "Content ideas" };
const KEYWORD = { field: "keyword", type: "text" };
const VOLUME = { field: "volume", type: "number" };
const DRAFT = {
  label: "Draft",
  library_agent_id: "lib-draft",
  graph_id: "graph-draft",
  graph_version: 3,
  inputs: { keyword: "keyword" },
};

function idea(id: string, keyword: string, volume: number, status: string) {
  return {
    id,
    values: { keyword, volume, status },
    pending: {},
    manual_fields: [],
    written_by: "workflow:keyword-ideas",
    revision: 2,
    created_at: "2026-11-20T09:00:00Z",
    updated_at: "2026-11-21T09:00:00Z",
  };
}

function page(): ResolvedPage {
  return {
    id: "page-1",
    title: "Content desk",
    revision: 1,
    has_previous: false,
    sections: [
      {
        kind: "collection_board",
        index: 0,
        ...IDEAS,
        group_by: "status",
        card_title: KEYWORD,
        card_fields: [VOLUME],
        read_only: false,
        buttons: [DRAFT],
        columns: [
          {
            option: "New",
            total: 1,
            rows: [idea("i1", "cohort retention chart", 1900, "New")],
          },
          { option: "Drafted", total: 0, rows: [] },
          {
            option: "Posted",
            total: 1,
            rows: [idea("i4", "cohort churn formula", 390, "Posted")],
          },
        ],
      },
      {
        kind: "collection_cards",
        index: 1,
        ...IDEAS,
        title: "Biggest ideas",
        card_title: KEYWORD,
        card_fields: [VOLUME],
        rows: [idea("i1", "cohort retention chart", 1900, "New")],
        total: 4,
        buttons: [],
      },
      {
        kind: "collection_chart",
        index: 2,
        ...IDEAS,
        title: "Search volume by status",
        block: null,
      },
    ],
  } as unknown as ResolvedPage;
}

function serve() {
  const edits: { row: string; body: unknown }[] = [];
  server.use(
    http.get("/api/proxy/api/pages/:pageID", () => HttpResponse.json(page())),
    http.patch(
      "/api/proxy/api/collections/ideas/rows/:rowID",
      async ({ request, params }) => {
        edits.push({ row: String(params.rowID), body: await request.json() });
        return HttpResponse.json(idea("i1", "x", 1, "Drafted"));
      },
    ),
  );
  return edits;
}

test("dragging a card to another column sets the field the board groups by", async () => {
  const edits = serve();
  render(<PageViewPage />);

  const newColumn = await screen.findByRole("region", { name: "New" });
  const card = within(newColumn).getByText("cohort retention chart");
  fireEvent.dragStart(card.closest("[draggable]")!);
  const posted = screen.getByRole("region", { name: "Posted" });
  fireEvent.dragOver(posted);
  fireEvent.drop(posted);

  await waitFor(() => expect(edits).toHaveLength(1));
  expect(edits[0]).toEqual({
    row: "i1",
    body: { values: { status: "Posted" }, based_on: 2 },
  });
});

test("dropping a card in its own column writes nothing", async () => {
  const edits = serve();
  render(<PageViewPage />);

  const newColumn = await screen.findByRole("region", { name: "New" });
  fireEvent.dragStart(
    within(newColumn)
      .getByText("cohort retention chart")
      .closest("[draggable]")!,
  );
  fireEvent.dragOver(newColumn);
  fireEvent.drop(newColumn);

  await new Promise((resolve) => setTimeout(resolve, 50));
  expect(edits).toEqual([]);
});

test("a card's button runs the workflow with that row's value and rereads when it ends", async () => {
  serve();
  const started: unknown[] = [];
  let pageReads = 0;
  server.use(
    http.get("/api/proxy/api/pages/:pageID", () => {
      pageReads += 1;
      return HttpResponse.json(page());
    }),
    http.post(
      "/api/proxy/api/graphs/graph-draft/execute/3",
      async ({ request }) => {
        started.push(await request.json());
        return HttpResponse.json({ id: "run-1", graph_id: "graph-draft" });
      },
    ),
    http.get("/api/proxy/api/graphs/graph-draft/executions/run-1", () =>
      HttpResponse.json({ id: "run-1", status: "COMPLETED" }),
    ),
  );
  render(<PageViewPage />);

  await userEvent.click(
    await screen.findByRole("button", {
      name: "Draft: cohort retention chart",
    }),
  );

  await waitFor(() =>
    expect(started).toEqual([
      { inputs: { keyword: "cohort retention chart" } },
    ]),
  );
  const readsAfterStart = pageReads;
  await waitFor(() => expect(pageReads).toBeGreaterThan(readsAfterStart), {
    timeout: 8_000,
  });
});

test("cards show their fields and a chart with nothing to show says so", async () => {
  serve();
  render(<PageViewPage />);

  const cards = await screen.findByText("Biggest ideas");
  expect(cards).toBeDefined();
  expect(screen.getByText("1 of 4 rows")).toBeDefined();
  expect(screen.getByText("Nothing to chart yet.")).toBeDefined();
});
