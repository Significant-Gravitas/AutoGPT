import { beforeEach, describe, expect, test, vi } from "vitest";

import {
  getGetV2GetMyAgentsMockHandler,
  getGetV2ListMySubmissionsMockHandler,
  getGetV2ListMySubmissionsResponseMock,
} from "@/app/api/__generated__/endpoints/store/store.msw";
import type { MyUnpublishedAgent } from "@/app/api/__generated__/models/myUnpublishedAgent";
import type { MyUnpublishedAgentsResponse } from "@/app/api/__generated__/models/myUnpublishedAgentsResponse";
import type { StoreSubmissionsResponse } from "@/app/api/__generated__/models/storeSubmissionsResponse";
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

function makeAgent(
  overrides: Partial<MyUnpublishedAgent> = {},
): MyUnpublishedAgent {
  return {
    graph_id: FLOW_ID,
    graph_version: FLOW_VERSION,
    agent_name: AGENT_NAME,
    description: "An agent created in the builder",
    last_edited: new Date("2026-04-01T00:00:00Z"),
    ...overrides,
  };
}

function makeAgentsResponse(
  agents: MyUnpublishedAgent[],
): MyUnpublishedAgentsResponse {
  return {
    agents,
    pagination: {
      total_items: agents.length,
      total_pages: 1,
      current_page: 1,
      page_size: agents.length || 20,
    },
  };
}

function makeEmptySubmissionsResponse(): StoreSubmissionsResponse {
  return {
    submissions: [],
    pagination: {
      total_items: 0,
      total_pages: 1,
      current_page: 1,
      page_size: 20,
    },
    stats: {
      total: 0,
      approved: 0,
      pending: 0,
      total_runs: 0,
      average_rating: null,
    },
  };
}

function installScopedHandlers() {
  server.use(
    getGetV2GetMyAgentsMockHandler(makeAgentsResponse([makeAgent()])),
    getGetV2ListMySubmissionsMockHandler(makeEmptySubmissionsResponse()),
  );
}

const LIVE_DESCRIPTION = "Finds the decision-makers at any company.";

function installListedAgentHandlers() {
  const [listing] = getGetV2ListMySubmissionsResponseMock().submissions;
  server.use(
    getGetV2GetMyAgentsMockHandler(makeAgentsResponse([makeAgent()])),
    getGetV2ListMySubmissionsMockHandler(
      getGetV2ListMySubmissionsResponseMock({
        submissions: [
          {
            ...listing,
            graph_id: FLOW_ID,
            status: SubmissionStatus.APPROVED,
            description: LIVE_DESCRIPTION,
            changes_summary: "Initial Submission",
          },
        ],
      }),
    ),
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

  test("opens an update to a listed workflow with the live description, and says leaving it empty keeps it", async () => {
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
