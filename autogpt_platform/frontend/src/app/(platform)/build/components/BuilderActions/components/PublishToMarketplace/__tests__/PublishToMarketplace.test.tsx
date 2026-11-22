import { beforeEach, describe, expect, test, vi } from "vitest";

import {
  getGetV1GetSpecificGraphMockHandler,
  getGetV1GetSpecificGraphResponseMock,
} from "@/app/api/__generated__/endpoints/graphs/graphs.msw";
import {
  getGetV2GetLibraryAgentByGraphIdMockHandler,
  getGetV2GetLibraryAgentByGraphIdResponseMock,
} from "@/app/api/__generated__/endpoints/library/library.msw";
import {
  getGetV2GetMyLiveSubmissionMockHandler,
  getGetV2ListMySubmissionsResponseMock,
} from "@/app/api/__generated__/endpoints/store/store.msw";
import { SubmissionStatus } from "@/app/api/__generated__/models/submissionStatus";
import { server } from "@/mocks/mock-server";
import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";

const mockUseAuth = vi.hoisted(() => vi.fn());

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: mockUseAuth,
}));

vi.mock("@sentry/nextjs", async () => {
  const actual =
    await vi.importActual<typeof import("@sentry/nextjs")>("@sentry/nextjs");
  return {
    ...actual,
    captureException: vi.fn(),
    getTraceData: vi.fn(() => ({})),
    withServerActionInstrumentation: vi.fn((_, __, callback) => callback()),
  };
});

vi.mock("@/components/contextual/CronScheduler/cron-scheduler-dialog", () => ({
  CronExpressionDialog: () => null,
}));

vi.mock(
  "@/components/contextual/PublishAgentModal/components/AgentInfoStep/components/ThumbnailImages",
  () => ({
    ThumbnailImages: () => <div data-testid="thumbnail-images-mock" />,
  }),
);

import { PublishToMarketplace } from "../PublishToMarketplace";

const testUser = {
  id: "user-1",
  email: "user@example.com",
  app_metadata: {},
  user_metadata: {},
  aud: "authenticated",
  created_at: "2026-01-01T00:00:00.000Z",
};

const FLOW_ID = "graph-1";
const FLOW_VERSION = 3;
const AGENT_NAME = "My Builder Agent";

// The workflow open in the builder, read the way the dialog reads it: the
// graph at this version and its Library entry. It has no schedule of its own.
function serveBuilderWorkflow() {
  server.use(
    getGetV1GetSpecificGraphMockHandler(
      getGetV1GetSpecificGraphResponseMock({
        id: FLOW_ID,
        version: FLOW_VERSION,
        name: AGENT_NAME,
        description: "A workflow created in the builder",
        recommended_schedule_cron: null,
      }),
    ),
    getGetV2GetLibraryAgentByGraphIdMockHandler(
      getGetV2GetLibraryAgentByGraphIdResponseMock({
        graph_id: FLOW_ID,
        image_url: "https://example.com/builder-workflow.png",
      }),
    ),
  );
}

function installScopedHandlers() {
  serveBuilderWorkflow();
  server.use(getGetV2GetMyLiveSubmissionMockHandler(null));
}

const LIVE_DESCRIPTION = "Finds the decision-makers at any company.";

function installListedAgentHandlers() {
  const [listing] = getGetV2ListMySubmissionsResponseMock().submissions;
  serveBuilderWorkflow();
  server.use(
    getGetV2GetMyLiveSubmissionMockHandler({
      ...listing,
      graph_id: FLOW_ID,
      status: SubmissionStatus.APPROVED,
      description: LIVE_DESCRIPTION,
      changes_summary: "Initial Submission",
      recommended_schedule_cron: "0 9 * * 1",
    }),
  );
}

describe("PublishToMarketplace (builder)", () => {
  beforeEach(() => {
    mockUseAuth.mockReturnValue({
      user: testUser,
      isLoggedIn: true,
      isUserLoading: false,
    });
  });

  test("opens the listing form pre-scoped to the current agent and skips the picker", async () => {
    installScopedHandlers();

    render(
      <PublishToMarketplace flowID={FLOW_ID} flowVersion={FLOW_VERSION} />,
    );

    fireEvent.click(screen.getByRole("button"));

    expect(await screen.findByTestId("publish-agent-modal")).toBeDefined();
    expect(await screen.findByText("Build the store listing")).toBeDefined();

    expect(screen.queryByText(/choose a workflow/i)).toBeNull();

    expect(await screen.findByDisplayValue(AGENT_NAME)).toBeDefined();
  });

  test("opens an update to a listed workflow with the live description and schedule, and says leaving the description empty keeps it", async () => {
    installListedAgentHandlers();

    render(
      <PublishToMarketplace flowID={FLOW_ID} flowVersion={FLOW_VERSION} />,
    );

    fireEvent.click(screen.getByRole("button"));

    expect(await screen.findByText("Describe the update")).toBeDefined();
    fireEvent.click(screen.getByText("Experience details"));

    const description = await screen.findByRole("textbox", {
      name: "Description",
    });
    await waitFor(() => {
      expect((description as HTMLTextAreaElement).value).toBe(LIVE_DESCRIPTION);
    });
    const hintId = description.getAttribute("aria-describedby") ?? "";
    expect(document.getElementById(hintId)?.textContent).toBe(
      "Leave empty to keep the current description.",
    );
    expect(screen.getByText("Every Monday at 09:00")).toBeDefined();
    const changesSummary = screen.getByRole("textbox", {
      name: "What changed?",
    });
    expect((changesSummary as HTMLTextAreaElement).value).toBe("");
  });

  test("closes the modal when going back from the pre-scoped listing form", async () => {
    installScopedHandlers();

    render(
      <PublishToMarketplace flowID={FLOW_ID} flowVersion={FLOW_VERSION} />,
    );

    fireEvent.click(screen.getByRole("button"));

    expect(await screen.findByDisplayValue(AGENT_NAME)).toBeDefined();

    fireEvent.click(screen.getByRole("button", { name: /back/i }));

    await waitFor(() => {
      expect(screen.queryByTestId("publish-agent-modal")).toBeNull();
    });
  });

  test("disables the publish button when flowVersion is missing", () => {
    render(<PublishToMarketplace flowID={FLOW_ID} flowVersion={null} />);

    const button = screen.getByRole("button");
    expect((button as HTMLButtonElement).disabled).toBe(true);
  });

  test("disables the publish button when flowID is missing", () => {
    render(<PublishToMarketplace flowID={null} flowVersion={FLOW_VERSION} />);

    const button = screen.getByRole("button");
    expect((button as HTMLButtonElement).disabled).toBe(true);
  });
});
