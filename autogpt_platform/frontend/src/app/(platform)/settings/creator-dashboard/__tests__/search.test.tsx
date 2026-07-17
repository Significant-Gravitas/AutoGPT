import { beforeEach, describe, expect, test, vi } from "vitest";

import { getGetV2ListMySubmissionsMockHandler } from "@/app/api/__generated__/endpoints/store/store.msw";
import type { StoreSubmission } from "@/app/api/__generated__/models/storeSubmission";
import type { StoreSubmissionsResponse } from "@/app/api/__generated__/models/storeSubmissionsResponse";
import { SubmissionStatus } from "@/app/api/__generated__/models/submissionStatus";
import { server } from "@/mocks/mock-server";
import { within } from "@testing-library/react";

import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";

import SettingsCreatorDashboardPage from "../page";

const mockUseSupabase = vi.hoisted(() => vi.fn());

vi.mock("@/lib/supabase/hooks/useSupabase", () => ({
  useSupabase: mockUseSupabase,
}));

vi.mock("@/components/contextual/PublishAgentModal/PublishAgentModal", () => ({
  PublishAgentModal: ({ trigger }: { trigger: React.ReactNode }) => (
    <>{trigger}</>
  ),
}));

vi.mock("@/components/contextual/EditAgentModal/EditAgentModal", () => ({
  EditAgentModal: () => null,
}));

const testUser = {
  id: "user-1",
  email: "user@example.com",
  app_metadata: {},
  user_metadata: {},
  aud: "authenticated",
  created_at: "2026-01-01T00:00:00.000Z",
};

function makeSubmission(
  overrides: Partial<StoreSubmission> = {},
): StoreSubmission {
  return {
    listing_id: "listing-base",
    user_id: "user-1",
    slug: "agent-slug",
    listing_version_id: "lv-base",
    listing_version: 1,
    graph_id: "graph-base",
    graph_version: 1,
    name: "Base Agent",
    sub_heading: "sub",
    description: "desc",
    instructions: null,
    categories: [],
    image_urls: [],
    video_url: null,
    agent_output_demo_url: null,
    submitted_at: new Date("2026-04-01T00:00:00Z"),
    changes_summary: null,
    status: SubmissionStatus.PENDING,
    run_count: 0,
    review_count: 0,
    review_avg_rating: 0,
    ...overrides,
  };
}

function makeResponse(
  submissions: StoreSubmission[],
): StoreSubmissionsResponse {
  return {
    submissions,
    pagination: {
      total_items: submissions.length,
      total_pages: submissions.length > 0 ? 1 : 0,
      current_page: 1,
      page_size: submissions.length || 20,
    },
    stats: {
      total: submissions.length,
      approved: 0,
      pending: submissions.length,
      total_runs: 0,
      average_rating: null,
    },
  };
}

describe("creator-dashboard search", () => {
  beforeEach(() => {
    mockUseSupabase.mockReturnValue({
      user: testUser,
      isLoggedIn: true,
      isUserLoading: false,
      supabase: {},
    });
  });

  test("typing in search forwards a debounced search_query and renders matches", async () => {
    const observedQueries: (string | null)[] = [];
    server.use(
      getGetV2ListMySubmissionsMockHandler(({ request }) => {
        const url = new URL(request.url);
        const search = url.searchParams.get("search_query");
        observedQueries.push(search);
        const all = [
          makeSubmission({
            listing_version_id: "lv-1",
            name: "Invoice Agent",
          }),
          makeSubmission({
            listing_version_id: "lv-2",
            name: "Scraper Agent",
          }),
        ];
        const filtered = search
          ? all.filter((s) =>
              s.name.toLowerCase().includes(search.toLowerCase()),
            )
          : all;
        return makeResponse(filtered);
      }),
    );

    render(<SettingsCreatorDashboardPage />);

    const list = await screen.findByTestId("submissions-list");
    const scoped = within(list);

    expect(await scoped.findByText("Invoice Agent")).toBeDefined();
    expect(scoped.getByText("Scraper Agent")).toBeDefined();

    const searchBox = scoped.getByRole("searchbox", {
      name: /search submissions/i,
    });
    fireEvent.change(searchBox, { target: { value: "invoice" } });

    await waitFor(
      () => {
        expect(scoped.queryByText("Scraper Agent")).toBeNull();
      },
      { timeout: 2000 },
    );

    expect(scoped.getByText("Invoice Agent")).toBeDefined();
    expect(observedQueries.some((q) => q === "invoice")).toBe(true);
  });

  test("shows a search-specific empty state and clears via the empty-state button", async () => {
    server.use(
      getGetV2ListMySubmissionsMockHandler(({ request }) => {
        const url = new URL(request.url);
        const search = url.searchParams.get("search_query");
        if (search) return makeResponse([]);
        return makeResponse([
          makeSubmission({ listing_version_id: "lv-1", name: "Hello Agent" }),
        ]);
      }),
    );

    render(<SettingsCreatorDashboardPage />);

    const list = await screen.findByTestId("submissions-list");
    const scoped = within(list);

    expect(await scoped.findByText("Hello Agent")).toBeDefined();

    const searchBox = scoped.getByRole("searchbox", {
      name: /search submissions/i,
    });
    fireEvent.change(searchBox, { target: { value: "zzz" } });

    expect(
      await scoped.findByText(/No submissions match "zzz"/, undefined, {
        timeout: 2000,
      }),
    ).toBeDefined();

    const clearButtons = scoped.getAllByRole("button", {
      name: /clear search/i,
    });
    fireEvent.click(clearButtons[clearButtons.length - 1]);

    expect(await scoped.findByText("Hello Agent")).toBeDefined();
  });

  test("status filter forwards statuses to the submissions request", async () => {
    const observedStatuses: (string | null)[] = [];
    server.use(
      getGetV2ListMySubmissionsMockHandler(({ request }) => {
        const url = new URL(request.url);
        const statuses = url.searchParams.get("statuses");
        observedStatuses.push(statuses);

        const all = [
          makeSubmission({
            listing_version_id: "lv-1",
            name: "Pending Agent",
            status: SubmissionStatus.PENDING,
          }),
          makeSubmission({
            listing_version_id: "lv-2",
            name: "Approved Agent",
            status: SubmissionStatus.APPROVED,
          }),
        ];
        const selectedStatuses = statuses?.split(",") ?? [];
        const filtered =
          selectedStatuses.length > 0
            ? all.filter((submission) =>
                selectedStatuses.includes(submission.status),
              )
            : all;

        return makeResponse(filtered);
      }),
    );

    render(<SettingsCreatorDashboardPage />);

    expect(
      (await screen.findAllByText("Pending Agent")).length,
    ).toBeGreaterThan(0);

    fireEvent.click(
      screen.getAllByRole("button", { name: /filter status/i })[0],
    );
    fireEvent.click(
      await screen.findByRole("button", {
        name: /in review/i,
        pressed: false,
      }),
    );

    await waitFor(
      () => {
        expect(observedStatuses).toContain(SubmissionStatus.PENDING);
        expect(screen.queryByText("Approved Agent")).toBeNull();
      },
      { timeout: 3000 },
    );
  });

  test("sort filter forwards sorting to the submissions request", async () => {
    const observedSorts: { key: string | null; dir: string | null }[] = [];
    server.use(
      getGetV2ListMySubmissionsMockHandler(({ request }) => {
        const url = new URL(request.url);
        observedSorts.push({
          key: url.searchParams.get("sort_key"),
          dir: url.searchParams.get("sort_dir"),
        });

        return makeResponse([
          makeSubmission({
            listing_version_id: "lv-1",
            name: "Alpha",
            run_count: 100,
          }),
          makeSubmission({
            listing_version_id: "lv-2",
            name: "Beta",
            run_count: 500,
          }),
        ]);
      }),
    );

    render(<SettingsCreatorDashboardPage />);

    expect((await screen.findAllByText("Alpha")).length).toBeGreaterThan(0);

    fireEvent.click(screen.getAllByRole("button", { name: /filter sort/i })[1]);
    fireEvent.click(
      await screen.findByRole("button", { name: /lowest first/i }),
    );

    await waitFor(
      () => {
        expect(observedSorts).toContainEqual({ key: "runs", dir: "asc" });
      },
      { timeout: 3000 },
    );
  });
});
