import type { ResolvedPage } from "@/app/api/__generated__/models/resolvedPage";
import { server } from "@/mocks/mock-server";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { beforeEach, expect, test, vi } from "vitest";
import PageViewPage from "../[id]/page";
import PagesListPage from "../page";

const params = vi.hoisted(() => ({ search: "" }));

vi.mock("next/navigation", async (importOriginal) => ({
  ...(await importOriginal<typeof import("next/navigation")>()),
  useParams: () => ({ id: "page-1" }),
  usePathname: () => "/pages/page-1",
  useRouter: () => ({ push: vi.fn(), replace: vi.fn(), refresh: vi.fn() }),
  useSearchParams: () => new URLSearchParams(params.search),
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

beforeEach(() => {
  params.search = "";
});

const LEDGER = { collection_id: "col-1", collection_name: "Invoices" };
const COLUMNS = [
  { field: "invoice_no", type: "text" },
  { field: "amount", type: "number", unit: "$" },
  { field: "status", type: "select", options: ["Booked", "Needs review"] },
];

function row(id: string, invoiceNo: string, amount: number, status: string) {
  return {
    id,
    values: { invoice_no: invoiceNo, amount, status },
    pending: {},
    manual_fields: [],
    written_by: "workflow:intake",
    revision: 1,
    created_at: "2026-11-21T09:00:00Z",
    updated_at: "2026-11-21T09:00:00Z",
  };
}

function page(): ResolvedPage {
  return {
    id: "page-1",
    title: "Invoices",
    revision: 1,
    has_previous: false,
    sections: [
      {
        kind: "collection_numbers",
        index: 0,
        ...LEDGER,
        block: {
          type: "stats",
          items: [
            { label: "Booked", value: "$1,820.50" },
            { label: "Needs review", value: "1" },
          ],
        },
      },
      {
        kind: "collection_table",
        index: 1,
        ...LEDGER,
        title: "Ledger",
        columns: COLUMNS,
        dropped_columns: ["approved_by"],
        rows: [
          row("r1", "INV-2003", 3400, "Needs review"),
          row("r2", "INV-2002", 1180.5, "Booked"),
        ],
        total: 2,
        filter: [],
        read_only: false,
      },
      {
        kind: "collection_table",
        index: 2,
        ...LEDGER,
        title: "Read only",
        columns: [COLUMNS[0]],
        rows: [row("r3", "INV-2001", 640, "Booked")],
        total: 1,
        filter: [],
        read_only: true,
      },
    ],
  } as unknown as ResolvedPage;
}

test("a page shows a collection's numbers and its table, and says what it lost", async () => {
  server.use(
    http.get("/api/proxy/api/pages/:pageID", () => HttpResponse.json(page())),
  );

  render(<PageViewPage />);

  expect(await screen.findByText("$1,820.50")).toBeDefined();
  expect(screen.getByText("INV-2003")).toBeDefined();
  expect(screen.getByText("$3,400")).toBeDefined();
  expect(
    screen.getByText("No longer in the collection: approved_by"),
  ).toBeDefined();
  expect(
    screen.getByRole("button", { name: "Edit invoice_no of INV-2003" }),
  ).toBeDefined();
  expect(
    screen.queryByRole("button", { name: "Edit invoice_no of INV-2001" }),
  ).toBeNull();
});

test("searching and flipping a chip ask the rows route for exactly that", async () => {
  const asked: URLSearchParams[] = [];
  server.use(
    http.get("/api/proxy/api/pages/:pageID", () => HttpResponse.json(page())),
    http.get("/api/proxy/api/collections/col-1/rows", ({ request }) => {
      const query = new URL(request.url).searchParams;
      asked.push(query);
      const narrowed = query.get("search") === "nordkai";
      return HttpResponse.json({
        rows: narrowed ? [row("r2", "INV-2002", 1180.5, "Booked")] : [],
        total: narrowed ? 1 : 0,
      });
    }),
  );
  render(<PageViewPage />);

  await userEvent.type(
    (await screen.findAllByLabelText("Search rows"))[0],
    "nordkai",
  );

  await waitFor(() => expect(screen.queryByText("INV-2003")).toBeNull());
  expect(screen.getByText("INV-2002")).toBeDefined();
  await userEvent.click(
    screen.getAllByRole("button", { name: "Needs review" })[0],
  );
  await waitFor(() =>
    expect(
      asked.some(
        (q) =>
          q.get("where") ===
          JSON.stringify({
            conditions: [],
            chips: { status: ["Needs review"] },
          }),
      ),
    ).toBe(true),
  );
});

test("the Collections tab lists each collection with its rows and fields", async () => {
  params.search = "tab=collections";
  server.use(
    http.get("/api/proxy/api/pages", () => HttpResponse.json({ pages: [] })),
    http.get("/api/proxy/api/collections", () =>
      HttpResponse.json({
        collections: [
          {
            id: "col-1",
            name: "Invoices",
            fields: [
              { name: "invoice_no", type: "text" },
              { name: "amount", type: "number" },
            ],
            revision: 1,
            row_count: 12,
            updated_at: "2026-11-21T12:00:00Z",
          },
        ],
        limits: { collections: 50, rows: 5000, fields: 30, row_bytes: 8192 },
      }),
    ),
  );

  render(<PagesListPage />);

  const link = await screen.findByRole("link", { name: /Invoices/ });
  expect(link.getAttribute("href")).toBe("/collections/col-1");
  expect(link.textContent).toContain("12 rows · 2 fields");
});

test("with no collections the tab says how one is made", async () => {
  params.search = "tab=collections";
  server.use(
    http.get("/api/proxy/api/pages", () => HttpResponse.json({ pages: [] })),
    http.get("/api/proxy/api/collections", () =>
      HttpResponse.json({
        collections: [],
        limits: { collections: 50, rows: 5000, fields: 30, row_bytes: 8192 },
      }),
    ),
  );

  render(<PagesListPage />);

  expect(await screen.findByText("No collections yet")).toBeDefined();
});
