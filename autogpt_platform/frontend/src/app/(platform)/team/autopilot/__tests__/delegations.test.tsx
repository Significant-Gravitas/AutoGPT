import {
  getListDelegationsMockHandler200,
  getListDelegationsMockHandler422,
} from "@/app/api/__generated__/endpoints/experts/experts.msw";
import { server } from "@/mocks/mock-server";
import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, test, vi } from "vitest";
import { NuqsTestingAdapter } from "nuqs/adapters/testing";
import AutopilotPage from "../page";
import {
  delegation,
  listDelegationsLikeTheServer,
  mockOttoApi,
} from "./delegationFixtures";

// The tabs live in ?tab=; a remembering adapter keeps a clicked tab open.
function renderPage() {
  return render(
    <NuqsTestingAdapter hasMemory>
      <AutopilotPage />
    </NuqsTestingAdapter>,
  );
}

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return {
    ...actual,
    useFlagStatus: () => ({ enabled: true, ready: true }),
  };
});

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn(), prefetch: vi.fn() }),
  usePathname: () => "/team/autopilot",
  useSearchParams: () => new URLSearchParams(),
  useParams: () => ({}),
  notFound: () => {
    throw new Error("NEXT_NOT_FOUND");
  },
}));

beforeEach(() => {
  mockOttoApi();
});

async function openTab(name: string) {
  await userEvent.click(await screen.findByRole("tab", { name }));
}

describe("Otto's Delegations tab", () => {
  test("sums up today's hand-offs and lists every one with its route and status", async () => {
    renderPage();
    expect(await screen.findByText("3 delegations today")).toBeDefined();

    await openTab("Delegations");

    expect((await screen.findByTestId("delegations-summary")).textContent).toBe(
      "3 delegations today · 1 working · 1 needs you · $0.40 spent",
    );
    expect(screen.getByText("Ask first")).toBeDefined();

    const list = screen.getByRole("list", { name: "Delegations" });
    const rows = within(list).getAllByRole("listitem");
    expect(rows).toHaveLength(4);
    expect(
      within(rows[0]).getByText("Otto → Devon · waiting for your approval"),
    ).toBeDefined();
    expect(within(rows[0]).getByText("Waiting for you")).toBeDefined();
    expect(
      within(rows[1]).getByText("Otto → Devon · working · $0.04"),
    ).toBeDefined();
    expect(
      within(rows[2]).getByText("Otto → Alex · 6m 40s · $0.31 · 1 file"),
    ).toBeDefined();
    expect(within(rows[2]).getByText("Done")).toBeDefined();
    expect(within(rows[3]).getByText("Failed")).toBeDefined();
    expect(within(rows[0]).getByRole("link").getAttribute("href")).toBe(
      "/copilot?sessionId=otto-chat",
    );
  });

  test("filters by what needs you, what is working and what failed", async () => {
    renderPage();
    await openTab("Delegations");
    const filters = await screen.findByRole("group", {
      name: "Filter delegations",
    });

    function rowTexts() {
      return within(screen.getByRole("list", { name: "Delegations" }))
        .getAllByRole("listitem")
        .map((row) => row.textContent);
    }

    fireEvent.click(within(filters).getByRole("button", { name: "Needs you" }));
    await waitFor(() =>
      expect(rowTexts()).toEqual([
        expect.stringContaining("Retention policy check"),
      ]),
    );

    fireEvent.click(within(filters).getByRole("button", { name: "Working" }));
    await waitFor(() =>
      expect(rowTexts()).toEqual([
        expect.stringContaining("Eng estimate for onboarding revamp"),
      ]),
    );

    fireEvent.click(within(filters).getByRole("button", { name: "Failed" }));
    expect(await screen.findByText("Partner outreach list")).toBeDefined();
    expect(screen.queryByText("Onboarding revamp PRD")).toBeNull();
  });

  test("finds an old hand-off waiting on the user, and shows more on request", async () => {
    const recent = Array.from({ length: 55 }, (_, i) =>
      delegation({
        sub_session_id: `done-${i}`,
        title: `Finished task ${i}`,
        status: "completed",
      }),
    );
    const waiting = delegation({
      sub_session_id: "old-question",
      title: "Pricing page copy",
      status: "needs_input",
      question: "Annual or monthly first?",
    });
    server.use(
      listDelegationsLikeTheServer([...recent, waiting], {
        working: 0,
        needs_you: 1,
        completed: 55,
        failed: 0,
        started_today: 56,
        spent_today_usd: 0,
      }),
    );
    renderPage();
    await openTab("Delegations");
    const list = await screen.findByRole("list", { name: "Delegations" });
    expect(within(list).getAllByRole("listitem")).toHaveLength(50);
    expect(screen.queryByText("Pricing page copy")).toBeNull();

    // Past the first page, yet Needs you finds it: the server filters.
    const filters = screen.getByRole("group", { name: "Filter delegations" });
    fireEvent.click(within(filters).getByRole("button", { name: "Needs you" }));
    expect(await screen.findByText("Pricing page copy")).toBeDefined();

    fireEvent.click(within(filters).getByRole("button", { name: "All" }));
    fireEvent.click(await screen.findByRole("button", { name: "Show more" }));
    await waitFor(() =>
      expect(
        within(screen.getByRole("list", { name: "Delegations" })).getAllByRole(
          "listitem",
        ),
      ).toHaveLength(56),
    );
    expect(screen.queryByRole("button", { name: "Show more" })).toBeNull();
    expect(screen.getByText("56 delegations today")).toBeDefined();
  });

  test("says so when Otto has not handed anything off", async () => {
    server.use(
      getListDelegationsMockHandler200({
        delegations: [],
        total: 0,
        summary: {
          working: 0,
          needs_you: 0,
          completed: 0,
          failed: 0,
          spent_today_usd: 0,
        },
      }),
    );
    renderPage();
    await openTab("Delegations");

    expect(await screen.findByText(/No hand-offs yet/)).toBeDefined();
    expect(screen.queryByText(/delegations today$/)).toBeNull();
  });

  test("offers a retry when the hand-offs fail to load", async () => {
    server.use(getListDelegationsMockHandler422());
    renderPage();
    await openTab("Delegations");

    expect(
      await screen.findByText("We could not load the hand-offs."),
    ).toBeDefined();
  });

  test("Change opens the Settings tab", async () => {
    renderPage();
    await openTab("Delegations");
    await userEvent.click(
      await screen.findByRole("button", { name: "Change" }),
    );

    expect(await screen.findByText("Delegation mode")).toBeDefined();
  });
});
