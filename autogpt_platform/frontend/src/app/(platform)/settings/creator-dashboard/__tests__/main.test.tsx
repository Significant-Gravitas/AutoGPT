import { beforeEach, describe, expect, test, vi } from "vitest";

import {
  getDeleteV2DeleteStoreSubmissionMockHandler,
  getDeleteV2DeleteStoreSubmissionMockHandler422,
  getGetV2ListMySubmissionsMockHandler,
  getGetV2ListMySubmissionsMockHandler401,
} from "@/app/api/__generated__/endpoints/store/store.msw";
import type { StoreSubmission } from "@/app/api/__generated__/models/storeSubmission";
import type { StoreSubmissionsResponse } from "@/app/api/__generated__/models/storeSubmissionsResponse";
import { SubmissionStatus } from "@/app/api/__generated__/models/submissionStatus";
import { server } from "@/mocks/mock-server";
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
  EditAgentModal: ({
    isOpen,
    onSuccess,
  }: {
    isOpen: boolean;
    onSuccess: (s: unknown) => void;
  }) =>
    isOpen ? (
      <button
        type="button"
        data-testid="edit-modal-success"
        onClick={() =>
          onSuccess({
            listing_id: "listing-base",
            user_id: "user-1",
            slug: "agent-slug",
            listing_version_id: "lv-edited",
            listing_version: 2,
            graph_id: "graph-base",
            graph_version: 2,
            name: "Edited Agent",
            sub_heading: "sub",
            description: "desc",
            instructions: null,
            categories: [],
            image_urls: [],
            video_url: null,
            agent_output_demo_url: null,
            submitted_at: null,
            changes_summary: null,
            status: SubmissionStatus.PENDING,
          })
        }
      >
        success
      </button>
    ) : null,
}));

const toastSpy = vi.hoisted(() => vi.fn());
vi.mock("@/components/molecules/Toast/use-toast", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/components/molecules/Toast/use-toast")
    >();
  return {
    ...actual,
    toast: (...args: Parameters<typeof actual.toast>) => toastSpy(...args),
  };
});

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
      total_pages: 1,
      current_page: 1,
      page_size: submissions.length || 20,
    },
  };
}

describe("SettingsCreatorDashboardPage", () => {
  beforeEach(() => {
    toastSpy.mockClear();
    mockUseSupabase.mockReturnValue({
      user: testUser,
      isLoggedIn: true,
      isUserLoading: false,
      supabase: {},
    });
  });

  test("renders header, stats, and submission rows from the API", async () => {
    server.use(
      getGetV2ListMySubmissionsMockHandler(
        makeResponse([
          makeSubmission({
            listing_version_id: "lv-1",
            name: "Approved Agent",
            status: SubmissionStatus.APPROVED,
            run_count: 1500,
            review_avg_rating: 4.2,
          }),
          makeSubmission({
            listing_version_id: "lv-2",
            name: "Pending Agent",
            status: SubmissionStatus.PENDING,
            run_count: 50,
            review_avg_rating: 0,
          }),
        ]),
      ),
    );

    render(<SettingsCreatorDashboardPage />);

    expect(
      await screen.findByRole("heading", { name: /creator dashboard/i }),
    ).toBeDefined();

    expect(
      await screen.findByRole("region", { name: /submission stats/i }),
    ).toBeDefined();

    expect(screen.getByText("Total submissions")).toBeDefined();
    expect(screen.getByText("Total runs")).toBeDefined();
    expect(screen.getAllByText("Approved").length).toBeGreaterThan(0);
    expect(screen.getAllByText("In review").length).toBeGreaterThan(0);

    expect(
      (await screen.findAllByText("Approved Agent")).length,
    ).toBeGreaterThan(0);
    expect(screen.getAllByText("Pending Agent").length).toBeGreaterThan(0);

    expect(screen.getAllByText("1.5K").length).toBeGreaterThan(0);
  });

  test("renders empty state when API returns no submissions", async () => {
    server.use(getGetV2ListMySubmissionsMockHandler(makeResponse([])));

    render(<SettingsCreatorDashboardPage />);

    expect(await screen.findByText(/no submissions yet/i)).toBeDefined();

    expect(
      screen.queryByRole("region", { name: /submission stats/i }),
    ).toBeNull();
    expect(screen.queryByTestId("submissions-list")).toBeNull();
  });

  test("renders error card when the submissions API fails", async () => {
    server.use(getGetV2ListMySubmissionsMockHandler401());

    render(<SettingsCreatorDashboardPage />);

    expect(await screen.findByText(/something went wrong/i)).toBeDefined();
    expect(
      screen.queryByRole("heading", { name: /creator dashboard/i }),
    ).toBeNull();
  });

  test("error card 'Try again' refetches and recovers", async () => {
    server.use(getGetV2ListMySubmissionsMockHandler401());

    render(<SettingsCreatorDashboardPage />);

    await screen.findByText(/something went wrong/i);

    server.use(
      getGetV2ListMySubmissionsMockHandler(
        makeResponse([
          makeSubmission({
            listing_version_id: "lv-recover",
            name: "Recovered Agent",
            status: SubmissionStatus.PENDING,
          }),
        ]),
      ),
    );

    fireEvent.click(screen.getByRole("button", { name: /try again/i }));

    expect(
      (await screen.findAllByText("Recovered Agent")).length,
    ).toBeGreaterThan(0);
  });

  test("renders both pending and approved submissions in the list", async () => {
    server.use(
      getGetV2ListMySubmissionsMockHandler(
        makeResponse([
          makeSubmission({
            listing_version_id: "lv-1",
            name: "Approved Agent",
            status: SubmissionStatus.APPROVED,
          }),
          makeSubmission({
            listing_version_id: "lv-2",
            name: "Pending Agent",
            status: SubmissionStatus.PENDING,
          }),
        ]),
      ),
    );

    render(<SettingsCreatorDashboardPage />);

    expect(
      (await screen.findAllByText("Approved Agent")).length,
    ).toBeGreaterThan(0);

    expect(screen.getAllByText("Pending Agent").length).toBeGreaterThan(0);
  });

  test("deleting a pending submission via row action calls the delete endpoint", async () => {
    let deletedId: string | null = null;

    server.use(
      getGetV2ListMySubmissionsMockHandler(
        makeResponse([
          makeSubmission({
            listing_version_id: "lv-target",
            name: "Deletable Agent",
            status: SubmissionStatus.PENDING,
          }),
        ]),
      ),
      getDeleteV2DeleteStoreSubmissionMockHandler(async ({ params }) => {
        deletedId = params.submissionId as string;
        return true;
      }),
    );

    render(<SettingsCreatorDashboardPage />);

    expect(
      (await screen.findAllByText("Deletable Agent")).length,
    ).toBeGreaterThan(0);

    const actionButtons = screen.getAllByTestId("submission-actions");
    fireEvent.pointerDown(actionButtons[0], { button: 0 });

    const deleteMenuItem = await screen.findByRole("menuitem", {
      name: /delete/i,
    });
    fireEvent.click(deleteMenuItem);

    const confirmButton = await screen.findByRole("button", {
      name: /delete submission/i,
    });
    fireEvent.click(confirmButton);

    await waitFor(() => {
      expect(deletedId).toBe("lv-target");
    });
  });

  test("approved submissions cannot be selected (no checkbox rendered)", async () => {
    server.use(
      getGetV2ListMySubmissionsMockHandler(
        makeResponse([
          makeSubmission({
            listing_version_id: "lv-approved",
            name: "Approved Agent",
            status: SubmissionStatus.APPROVED,
          }),
        ]),
      ),
    );

    render(<SettingsCreatorDashboardPage />);

    expect(
      (await screen.findAllByText("Approved Agent")).length,
    ).toBeGreaterThan(0);

    expect(
      screen.queryByRole("checkbox", { name: /select approved agent/i }),
    ).toBeNull();
  });

  test("delete failure shows a destructive toast and keeps the dialog open", async () => {
    server.use(
      getGetV2ListMySubmissionsMockHandler(
        makeResponse([
          makeSubmission({
            listing_version_id: "lv-fail",
            name: "Doomed Agent",
            status: SubmissionStatus.PENDING,
          }),
        ]),
      ),
      getDeleteV2DeleteStoreSubmissionMockHandler422(),
    );

    render(<SettingsCreatorDashboardPage />);

    expect((await screen.findAllByText("Doomed Agent")).length).toBeGreaterThan(
      0,
    );

    const actionButtons = screen.getAllByTestId("submission-actions");
    fireEvent.pointerDown(actionButtons[0], { button: 0 });

    const deleteMenuItem = await screen.findByRole("menuitem", {
      name: /delete/i,
    });
    fireEvent.click(deleteMenuItem);

    const confirmButton = await screen.findByRole("button", {
      name: /delete submission/i,
    });
    fireEvent.click(confirmButton);

    await waitFor(() => {
      expect(toastSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          title: "Couldn't delete submission",
          variant: "destructive",
        }),
      );
    });
  });

  test("bulk delete failure shows a partial-success destructive toast", async () => {
    server.use(
      getGetV2ListMySubmissionsMockHandler(
        makeResponse([
          makeSubmission({
            listing_version_id: "lv-bulk",
            name: "Bulk Agent",
            status: SubmissionStatus.PENDING,
          }),
        ]),
      ),
      getDeleteV2DeleteStoreSubmissionMockHandler422(),
    );

    render(<SettingsCreatorDashboardPage />);

    const checkboxes = await screen.findAllByRole("checkbox", {
      name: /select bulk agent/i,
    });
    fireEvent.click(checkboxes[0]);

    const deleteSelected = await screen.findByRole("button", {
      name: /delete selected/i,
    });
    fireEvent.click(deleteSelected);

    const confirm = await screen.findAllByRole("button", {
      name: /delete selected/i,
    });
    fireEvent.click(confirm[confirm.length - 1]);

    await waitFor(() => {
      expect(toastSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          variant: "destructive",
        }),
      );
    });
  });

  test("editing a submission via the row dropdown opens and closes the edit modal on success", async () => {
    server.use(
      getGetV2ListMySubmissionsMockHandler(
        makeResponse([
          makeSubmission({
            listing_version_id: "lv-edit",
            name: "Editable Agent",
            status: SubmissionStatus.PENDING,
          }),
        ]),
      ),
    );

    render(<SettingsCreatorDashboardPage />);

    expect(
      (await screen.findAllByText("Editable Agent")).length,
    ).toBeGreaterThan(0);

    const actionButtons = screen.getAllByTestId("submission-actions");
    fireEvent.pointerDown(actionButtons[0], { button: 0 });
    const editItem = await screen.findByRole("menuitem", { name: /edit/i });
    fireEvent.click(editItem);

    const successTriggers = await screen.findAllByTestId("edit-modal-success");
    fireEvent.click(successTriggers[0]);

    await waitFor(() => {
      expect(screen.queryByTestId("edit-modal-success")).toBeNull();
    });
  });

  test("approved row dropdown shows a 'View submission' action that calls onView", async () => {
    server.use(
      getGetV2ListMySubmissionsMockHandler(
        makeResponse([
          makeSubmission({
            listing_version_id: "lv-view",
            name: "Viewable Agent",
            status: SubmissionStatus.APPROVED,
          }),
        ]),
      ),
    );

    render(<SettingsCreatorDashboardPage />);

    expect(
      (await screen.findAllByText("Viewable Agent")).length,
    ).toBeGreaterThan(0);

    const actionButtons = screen.getAllByTestId("submission-actions");
    fireEvent.pointerDown(actionButtons[0], { button: 0 });

    const viewItem = await screen.findByRole("menuitem", {
      name: /view submission/i,
    });
    expect(screen.queryByRole("menuitem", { name: /^delete$/i })).toBeNull();
    fireEvent.click(viewItem);
  });

  test("clicking Submit agent triggers the publish modal flow", async () => {
    server.use(getGetV2ListMySubmissionsMockHandler(makeResponse([])));

    render(<SettingsCreatorDashboardPage />);

    const submitButton = await screen.findByTestId("submit-agent-button");
    fireEvent.click(submitButton);

    expect(submitButton).toBeDefined();
  });

  test("mobile dropdown also exposes Delete and triggers the same delete endpoint", async () => {
    let deletedId: string | null = null;
    server.use(
      getGetV2ListMySubmissionsMockHandler(
        makeResponse([
          makeSubmission({
            listing_version_id: "lv-mobile",
            name: "Mobile Agent",
            status: SubmissionStatus.PENDING,
          }),
        ]),
      ),
      getDeleteV2DeleteStoreSubmissionMockHandler(async ({ params }) => {
        deletedId = params.submissionId as string;
        return true;
      }),
    );

    render(<SettingsCreatorDashboardPage />);

    expect((await screen.findAllByText("Mobile Agent")).length).toBeGreaterThan(
      1,
    );

    const actionButtons = screen.getAllByTestId("submission-actions");
    expect(actionButtons.length).toBeGreaterThanOrEqual(2);
    fireEvent.pointerDown(actionButtons[1], { button: 0 });

    const deleteMenuItem = await screen.findByRole("menuitem", {
      name: /delete/i,
    });
    fireEvent.click(deleteMenuItem);

    const confirmButton = await screen.findByRole("button", {
      name: /delete submission/i,
    });
    fireEvent.click(confirmButton);

    await waitFor(() => {
      expect(deletedId).toBe("lv-mobile");
    });
  });

  test("status column filter popover toggles a status and reflects active state", async () => {
    server.use(
      getGetV2ListMySubmissionsMockHandler(
        makeResponse([
          makeSubmission({
            listing_version_id: "lv-1",
            name: "Pending One",
            status: SubmissionStatus.PENDING,
          }),
          makeSubmission({
            listing_version_id: "lv-2",
            name: "Approved One",
            status: SubmissionStatus.APPROVED,
          }),
        ]),
      ),
    );

    render(<SettingsCreatorDashboardPage />);

    expect((await screen.findAllByText("Pending One")).length).toBeGreaterThan(
      0,
    );

    const filterTriggers = screen.getAllByRole("button", {
      name: /filter status/i,
    });
    fireEvent.click(filterTriggers[0]);

    const inReview = await screen.findByRole("button", {
      name: /in review/i,
      pressed: false,
    });
    fireEvent.click(inReview);

    await waitFor(() => {
      expect(
        screen.getByRole("button", { name: /in review/i, pressed: true }),
      ).toBeDefined();
    });

    const clearButton = await screen.findByRole("button", { name: /^clear$/i });
    fireEvent.click(clearButton);
  });

  test("sort column filter applies and clears a sort direction", async () => {
    server.use(
      getGetV2ListMySubmissionsMockHandler(
        makeResponse([
          makeSubmission({
            listing_version_id: "lv-a",
            name: "Alpha",
            status: SubmissionStatus.APPROVED,
            run_count: 100,
          }),
          makeSubmission({
            listing_version_id: "lv-b",
            name: "Beta",
            status: SubmissionStatus.APPROVED,
            run_count: 500,
          }),
        ]),
      ),
    );

    render(<SettingsCreatorDashboardPage />);

    expect((await screen.findAllByText("Alpha")).length).toBeGreaterThan(0);

    const sortTriggers = screen.getAllByRole("button", {
      name: /filter sort/i,
    });
    fireEvent.click(sortTriggers[0]);

    const newestFirst = await screen.findByRole("button", {
      name: /newest first/i,
    });
    fireEvent.click(newestFirst);

    await waitFor(() => {
      expect(
        screen.getByRole("button", {
          name: /newest first/i,
          pressed: true,
        }),
      ).toBeDefined();
    });

    const clearButton = await screen.findByRole("button", { name: /^clear$/i });
    fireEvent.click(clearButton);
  });

  test("selection bar exposes Select All and Deselect controls", async () => {
    server.use(
      getGetV2ListMySubmissionsMockHandler(
        makeResponse([
          makeSubmission({
            listing_version_id: "lv-p1",
            name: "Pending One",
            status: SubmissionStatus.PENDING,
          }),
          makeSubmission({
            listing_version_id: "lv-p2",
            name: "Pending Two",
            status: SubmissionStatus.PENDING,
          }),
        ]),
      ),
    );

    render(<SettingsCreatorDashboardPage />);

    const firstCheckboxes = await screen.findAllByRole("checkbox", {
      name: /select pending one/i,
    });
    fireEvent.click(firstCheckboxes[0]);

    const selectAll = await screen.findByRole("button", {
      name: /select all/i,
    });
    fireEvent.click(selectAll);

    await waitFor(() => {
      expect(screen.queryByRole("button", { name: /select all/i })).toBeNull();
    });

    const deselect = screen.getByRole("button", { name: /^deselect$/i });
    fireEvent.click(deselect);

    await waitFor(() => {
      expect(screen.queryByRole("button", { name: /^deselect$/i })).toBeNull();
    });
  });

  test("selecting a pending submission shows the bulk selection bar", async () => {
    server.use(
      getGetV2ListMySubmissionsMockHandler(
        makeResponse([
          makeSubmission({
            listing_version_id: "lv-pending",
            name: "Pending Agent",
            status: SubmissionStatus.PENDING,
          }),
        ]),
      ),
    );

    render(<SettingsCreatorDashboardPage />);

    const checkboxes = await screen.findAllByRole("checkbox", {
      name: /select pending agent/i,
    });
    fireEvent.click(checkboxes[0]);

    expect(
      await screen.findByRole("button", { name: /delete selected/i }),
    ).toBeDefined();
  });
});
