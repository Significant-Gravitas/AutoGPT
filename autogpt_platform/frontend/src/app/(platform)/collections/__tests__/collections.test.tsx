import type { CollectionRowRecord } from "@/app/api/__generated__/models/collectionRowRecord";
import { server } from "@/mocks/mock-server";
import {
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { expect, test, vi } from "vitest";
import CollectionViewPage from "../[id]/page";

const mockToast = vi.hoisted(() => vi.fn());

vi.mock("@/components/molecules/Toast/use-toast", async (importOriginal) => ({
  ...(await importOriginal<
    typeof import("@/components/molecules/Toast/use-toast")
  >()),
  useToast: () => ({ toast: mockToast, dismiss: vi.fn() }),
}));

vi.mock("next/navigation", async (importOriginal) => ({
  ...(await importOriginal<typeof import("next/navigation")>()),
  useParams: () => ({ id: "col-1" }),
  usePathname: () => "/collections/col-1",
  useRouter: () => ({ push: vi.fn(), replace: vi.fn(), refresh: vi.fn() }),
  useSearchParams: () => new URLSearchParams(),
}));

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => ({
  ...(await importOriginal<
    typeof import("@/services/feature-flags/use-get-flag")
  >()),
  useGetFlag: (flag: string) => flag === "collections",
  useFlagStatus: (flag: string) => ({
    enabled: flag === "collections",
    ready: true,
    answered: true,
  }),
}));

const BASE = "/api/proxy/api/collections/col-1";
const COLLECTION = {
  id: "col-1",
  name: "Follow-ups",
  fields: [
    { name: "what", type: "text" },
    { name: "due", type: "date" },
    { name: "done", type: "checkbox" },
  ],
  key_field: null,
  revision: 2,
  row_count: 2,
  updated_at: "2026-11-21T12:00:00Z",
};
const LIMITS = { collections: 50, rows: 5000, fields: 30, row_bytes: 8192 };

function row(
  id: string,
  values: Record<string, unknown>,
  extra: Partial<CollectionRowRecord> = {},
) {
  return {
    id,
    values,
    pending: {},
    manual_fields: [],
    written_by: "otto",
    revision: 3,
    created_at: "2026-11-21T09:00:00Z",
    updated_at: "2026-11-21T09:00:00Z",
    ...extra,
  };
}

const QUOTE = row("r1", {
  what: "Send Brightline the revised quote",
  due: "2026-11-27",
  done: false,
});
const KICKOFF = row(
  "r2",
  { what: "Book Oko Studio's kickoff", due: "2026-12-03", done: false },
  { pending: { due: "2026-12-04" }, written_by: "workflow:g1" },
);

function serve(writes: { method: string; url: string; body: unknown }[]) {
  async function record(request: Request) {
    const body = request.method === "DELETE" ? null : await request.json();
    writes.push({ method: request.method, url: request.url, body });
  }
  server.use(
    http.get(BASE, () =>
      HttpResponse.json({ collection: COLLECTION, limits: LIMITS }),
    ),
    http.get(`${BASE}/rows`, () =>
      HttpResponse.json({ rows: [QUOTE, KICKOFF], total: 2 }),
    ),
    http.patch(`${BASE}/rows/:rowID`, async ({ request }) => {
      await record(request);
      return HttpResponse.json(QUOTE);
    }),
    http.post(`${BASE}/rows`, async ({ request }) => {
      await record(request);
      return HttpResponse.json(QUOTE);
    }),
    http.post(`${BASE}/rows/:rowID/accept`, async ({ request }) => {
      await record(request);
      return HttpResponse.json(KICKOFF);
    }),
    http.delete(`${BASE}/rows/:rowID`, async ({ request }) => {
      await record(request);
      return new HttpResponse(null, { status: 204 });
    }),
  );
}

test("the plain view shows the fields, the limits and every row", async () => {
  serve([]);

  render(<CollectionViewPage />);

  expect(
    await screen.findByRole("heading", { name: "Follow-ups" }),
  ).toBeDefined();
  expect(
    screen.getByText("Holds up to 5,000 rows and 30 fields, 8 KB a row."),
  ).toBeDefined();
  expect(
    await screen.findByText("Send Brightline the revised quote"),
  ).toBeDefined();
  expect(screen.getByText("Book Oko Studio's kickoff")).toBeDefined();
});

test("editing a cell sends the value and the version it was edited on", async () => {
  const writes: { method: string; url: string; body: unknown }[] = [];
  serve(writes);
  render(<CollectionViewPage />);

  await userEvent.click(
    await screen.findByRole("button", {
      name: "Edit what of Send Brightline the revised quote",
    }),
  );
  const box = screen.getByLabelText(
    "what of Send Brightline the revised quote",
  );
  await userEvent.clear(box);
  await userEvent.type(box, "Send Brightline the final quote{Enter}");

  await waitFor(() =>
    expect(writes).toEqual([
      {
        method: "PATCH",
        url: expect.stringContaining("/rows/r1"),
        body: {
          values: { what: "Send Brightline the final quote" },
          based_on: 3,
        },
      },
    ]),
  );
});

test("ticking a box, adding, deleting and accepting each reach their route", async () => {
  const writes: { method: string; url: string; body: unknown }[] = [];
  serve(writes);
  render(<CollectionViewPage />);

  await userEvent.click(
    await screen.findByRole("checkbox", {
      name: "done of Send Brightline the revised quote",
    }),
  );
  await waitFor(() => expect(writes).toHaveLength(1));
  await userEvent.click(
    screen.getByRole("button", {
      name: "Use the newer value for due of Book Oko Studio's kickoff",
    }),
  );
  await waitFor(() => expect(writes).toHaveLength(2));
  await userEvent.click(
    screen.getByRole("button", { name: "Delete Book Oko Studio's kickoff" }),
  );
  await waitFor(() => expect(writes).toHaveLength(3));
  await userEvent.click(screen.getByRole("button", { name: "Add row" }));
  const form = screen.getByRole("form", { name: "New row" });
  await userEvent.type(
    within(form).getByLabelText("what"),
    "Chase the Nordkai invoice",
  );
  await userEvent.click(within(form).getByRole("button", { name: "Add row" }));

  await waitFor(() => expect(writes).toHaveLength(4));
  expect(writes.map((w) => [w.method, w.body])).toEqual([
    ["PATCH", { values: { done: true }, based_on: 3 }],
    ["POST", { fields: ["due"], based_on: 3 }],
    ["DELETE", null],
    ["POST", { values: { what: "Chase the Nordkai invoice" } }],
  ]);
  expect(
    screen.getByText(/The workflow has a newer value: 2026-12-04/),
  ).toBeDefined();
});

test("an edit on a row that changed since is refused and says so", async () => {
  serve([]);
  server.use(
    http.patch(`${BASE}/rows/:rowID`, () =>
      HttpResponse.json(
        {
          detail: "This row changed since you opened it",
          row: { ...QUOTE, values: { ...QUOTE.values, done: true } },
        },
        { status: 409 },
      ),
    ),
  );
  render(<CollectionViewPage />);

  await userEvent.click(
    await screen.findByRole("checkbox", {
      name: "done of Send Brightline the revised quote",
    }),
  );

  await waitFor(() =>
    expect(mockToast).toHaveBeenCalledWith(
      expect.objectContaining({
        title: "This row changed since you opened it",
      }),
    ),
  );
});
