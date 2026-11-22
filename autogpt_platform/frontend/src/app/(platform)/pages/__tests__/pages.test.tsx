import type { ResolvedPage } from "@/app/api/__generated__/models/resolvedPage";
import { server } from "@/mocks/mock-server";
import { COPILOT_PROMPT_KEYS } from "@/services/copilot/prompt-handoff";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { beforeEach, expect, test, vi } from "vitest";
import PageViewPage from "../[id]/page";
import { useFollowedRunStore } from "../components/Page/store";
import PagesListPage from "../page";

const router = vi.hoisted(() => ({ push: vi.fn() }));

vi.mock("next/navigation", async (importOriginal) => ({
  ...(await importOriginal<typeof import("next/navigation")>()),
  useParams: () => ({ id: "page-1" }),
  usePathname: () => "/pages/page-1",
  useRouter: () => ({ push: router.push, replace: vi.fn(), refresh: vi.fn() }),
  useSearchParams: () => new URLSearchParams(),
}));

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => ({
  ...(await importOriginal<
    typeof import("@/services/feature-flags/use-get-flag")
  >()),
  useGetFlag: (flag: string) => flag === "pages",
  useFlagStatus: (flag: string) => ({
    enabled: flag === "pages",
    ready: true,
    answered: true,
  }),
}));

beforeEach(() => {
  router.push.mockClear();
  useFollowedRunStore.setState({ runs: {} });
});

const EXPERT = { expert_id: "nadia", expert_name: "Nadia" };

function page(overrides: Partial<ResolvedPage> = {}): ResolvedPage {
  return {
    id: "page-1",
    title: "Competitors",
    revision: 2,
    pinned_library_agent_id: null,
    chat_session_id: "chat-1",
    has_previous: true,
    sections: [
      {
        kind: "expert_status",
        index: 0,
        ...EXPERT,
        status: "working",
        detail: "Running Competitor watch",
        spend_cents: 120,
      },
      {
        kind: "expert_learned",
        index: 1,
        ...EXPERT,
        facts: [{ fact: "Acme cut its prices by 10%.", created_at: null }],
      },
      {
        kind: "waiting",
        index: 2,
        scope_name: null,
        reviews: [
          {
            node_exec_id: "node-exec-1",
            user_id: "user-1",
            graph_exec_id: "exec-1",
            graph_id: "graph-1",
            graph_version: 1,
            payload: { amount: 2400 },
            instructions: "Approve this invoice",
            editable: false,
            status: "WAITING",
          },
        ],
      },
    ],
    ...overrides,
  } as ResolvedPage;
}

test("the list says how to get a first page, and asking opens a chat to fill in", async () => {
  server.use(
    http.get("/api/proxy/api/pages", () => HttpResponse.json({ pages: [] })),
  );

  render(<PagesListPage />);

  expect(await screen.findByText("No pages yet")).toBeDefined();
  await userEvent.click(
    screen.getByRole("button", { name: "Ask Otto for a page" }),
  );
  expect(router.push).toHaveBeenCalledWith("/copilot");
  expect(sessionStorage.getItem(COPILOT_PROMPT_KEYS.prompt)).toBe(
    "Make me a page that ",
  );
});

test("the list links each page and says what it draws on", async () => {
  server.use(
    http.get("/api/proxy/api/pages", () =>
      HttpResponse.json({
        pages: [
          {
            id: "page-1",
            title: "Competitors",
            revision: 2,
            workflow_count: 2,
            expert_count: 1,
            pinned_library_agent_id: "lib-1",
            updated_at: "2026-11-21T12:00:00Z",
          },
        ],
      }),
    ),
  );

  render(<PagesListPage />);

  const link = await screen.findByRole("link", { name: /Competitors/ });
  expect(link.getAttribute("href")).toBe("/pages/page-1");
  expect(link.textContent).toContain("2 workflows · 1 expert · pinned");
});

test("a page shows what its expert is doing, what it learned and what waits", async () => {
  server.use(
    http.get("/api/proxy/api/pages/:pageID", () => HttpResponse.json(page())),
  );

  render(<PageViewPage />);

  expect(await screen.findByText("Running Competitor watch")).toBeDefined();
  expect(screen.getByText("$1.20 this week")).toBeDefined();
  expect(screen.getByText("Acme cut its prices by 10%.")).toBeDefined();
  expect(
    (await screen.findAllByText("Approve this invoice")).length,
  ).toBeGreaterThan(0);
  expect(screen.getByRole("button", { name: "Approve" })).toBeDefined();
});

test("renaming and undoing a change reach their routes", async () => {
  const edits: unknown[] = [];
  const restored = vi.fn();
  server.use(
    http.get("/api/proxy/api/pages/:pageID", () => HttpResponse.json(page())),
    http.patch("/api/proxy/api/pages/:pageID", async ({ request }) => {
      edits.push(await request.json());
      return HttpResponse.json(page());
    }),
    http.post("/api/proxy/api/pages/:pageID/restore", () => {
      restored();
      return HttpResponse.json(page());
    }),
  );

  render(<PageViewPage />);

  await userEvent.click(
    await screen.findByRole("button", { name: "Page actions" }),
  );
  await userEvent.click(
    await screen.findByRole("menuitem", { name: "Rename" }),
  );
  const title = await screen.findByLabelText("Title");
  await userEvent.clear(title);
  await userEvent.type(title, "Rivals{Enter}");
  await waitFor(() => expect(edits).toEqual([{ title: "Rivals" }]));

  await userEvent.click(screen.getByRole("button", { name: "Page actions" }));
  await userEvent.click(
    await screen.findByRole("menuitem", { name: "Undo last change" }),
  );
  await waitFor(() => expect(restored).toHaveBeenCalledTimes(1));
});

test("another account's page reads as unavailable", async () => {
  server.use(
    http.get("/api/proxy/api/pages/:pageID", () =>
      HttpResponse.json({ detail: "Not found" }, { status: 404 }),
    ),
  );

  render(<PageViewPage />);

  expect(await screen.findByText(/This page isn.t available/)).toBeDefined();
  expect(screen.queryByTestId("page")).toBeNull();
});
