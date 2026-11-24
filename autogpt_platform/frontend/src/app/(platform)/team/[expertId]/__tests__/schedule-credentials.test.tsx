import {
  getGetV1GetSpecificGraphMockHandler,
  getGetV1GetSpecificGraphResponseMock,
} from "@/app/api/__generated__/endpoints/graphs/graphs.msw";
import {
  getGetV2GetLibraryAgentMockHandler,
  getGetV2GetLibraryAgentResponseMock,
} from "@/app/api/__generated__/endpoints/library/library.msw";
import {
  getPostV1CreateExecutionScheduleMockHandler,
  getPostV1CreateExecutionScheduleResponseMock,
} from "@/app/api/__generated__/endpoints/schedules/schedules.msw";
import { getListExpertCredentialsMockHandler } from "@/app/api/__generated__/endpoints/experts/experts.msw";
import { server } from "@/mocks/mock-server";
import CredentialsProvider from "@/providers/agent-credentials/credentials-provider";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { HttpResponse, http } from "msw";
import { expect, test, vi } from "vitest";
import { CreateScheduleDialog } from "../components/CreateScheduleDialog";

// CredentialsProvider only loads credentials for a signed-in user.
vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({
    user: { id: "user-1", email: "u@example.com" },
    isLoggedIn: true,
    isUserLoading: false,
  }),
}));

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => ({
  ...(await importOriginal<
    typeof import("@/services/feature-flags/use-get-flag")
  >()),
  useGetFlag: () => false,
  useFlagStatus: () => ({ enabled: false, ready: true }),
}));

const FIELD = "firecrawl_api_key_credentials";
const PLATFORM_KEY = {
  id: "firecrawl-default",
  provider: "firecrawl",
  type: "api_key",
  title: "Firecrawl API Key",
  is_system: true,
};
const USER_KEY = {
  id: "user-firecrawl-key",
  provider: "firecrawl",
  type: "api_key",
  title: "My Firecrawl key",
  is_system: false,
};

test("a schedule made for an expert's workflow starts on the platform key, not a step's key the expert was never granted", async () => {
  const scheduleBodies: Array<{ credentials?: unknown }> = [];
  server.use(
    getGetV2GetLibraryAgentMockHandler(
      getGetV2GetLibraryAgentResponseMock({
        id: "library-agent",
        graph_id: "graph-1",
        graph_version: 1,
        name: "Scrape agent",
        can_access_graph: true,
        input_schema: { type: "object", properties: {} },
        credentials_input_schema: {
          type: "object",
          properties: {
            [FIELD]: {
              credentials_provider: ["firecrawl"],
              credentials_types: ["api_key"],
              title: "Credentials",
              type: "object",
            },
          },
          required: [FIELD],
        },
        has_external_trigger: false,
        trigger_setup_info: null,
        has_sensitive_action: false,
        has_human_in_the_loop: false,
        recommended_schedule_cron: null,
      }),
    ),
    // The workflow's step is set to the user's own Firecrawl key
    getGetV1GetSpecificGraphMockHandler(
      getGetV1GetSpecificGraphResponseMock({
        id: "graph-1",
        version: 1,
        links: [],
        sub_graphs: [],
        nodes: [
          {
            id: "scrape-step",
            block_id: "firecrawl-scrape",
            graph_id: "graph-1",
            graph_version: 1,
            input_default: {
              url: "https://example.com",
              credentials: {
                id: USER_KEY.id,
                provider: "firecrawl",
                type: "api_key",
                title: USER_KEY.title,
              },
            },
            metadata: {},
            input_links: [],
            output_links: [],
          },
        ],
      }),
    ),
    // The expert was granted none of the user's credentials
    getListExpertCredentialsMockHandler([]),
    http.get("*/api/integrations/providers", () =>
      HttpResponse.json([{ name: "firecrawl", description: "firecrawl" }]),
    ),
    http.get("*/api/integrations/providers/system", () =>
      HttpResponse.json(["firecrawl"]),
    ),
    http.get("*/api/integrations/credentials", () =>
      HttpResponse.json([PLATFORM_KEY, USER_KEY]),
    ),
    getPostV1CreateExecutionScheduleMockHandler(async ({ request }) => {
      scheduleBodies.push((await request.json()) as never);
      return getPostV1CreateExecutionScheduleResponseMock({
        id: "new-schedule",
        graph_id: "graph-1",
      });
    }),
  );

  render(
    <CredentialsProvider>
      <CreateScheduleDialog
        expertId="expert-1"
        open
        onClose={vi.fn()}
        workflows={[
          {
            id: "workflow-1",
            store_listing_version_id: null,
            library_agent_id: "library-agent",
            graph_id: "graph-1",
            name: "Scrape agent",
            description: null,
          },
        ]}
      />
    </CredentialsProvider>,
  );

  // The row is re-rendered as the run form's trigger once the workflow loads
  await waitFor(() =>
    expect(
      screen
        .getByRole("button", { name: /^Scrape agent/ })
        .hasAttribute("disabled"),
    ).toBe(false),
  );
  await userEvent.click(screen.getByRole("button", { name: /^Scrape agent/ }));
  const scheduleTask = await screen.findByRole("button", {
    name: "Schedule Task",
  });
  await waitFor(() =>
    expect(scheduleTask.hasAttribute("disabled")).toBe(false),
  );
  await userEvent.click(scheduleTask);
  const schedule = await screen.findByRole("button", { name: "Schedule" });
  await waitFor(() => expect(schedule.hasAttribute("disabled")).toBe(false));
  await userEvent.click(schedule);

  await waitFor(() => expect(scheduleBodies).toHaveLength(1));
  // The backend attributes this schedule to the expert, and refuses every
  // run that names a credential the expert wasn't granted. The platform's
  // key needs no grant.
  expect(
    (scheduleBodies[0].credentials as Record<string, { id?: string }>)[FIELD]
      ?.id,
  ).toBe(PLATFORM_KEY.id);
});
