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
import { ExpertCredentialRef } from "@/app/api/__generated__/models/expertCredentialRef";
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
const PICKER = "Select Firecrawl credential";
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
const REFUSED =
  "This block keeps what it creates in the Firecrawl account behind its key, so it can't use the platform's shared Firecrawl key. Connect your own Firecrawl API key and select it instead.";

function keyIn(credentials: unknown) {
  return (credentials as Record<string, { id?: string }> | undefined)?.[FIELD]
    ?.id;
}

// An expert's workflow whose step is set to the user's own Firecrawl key
function mockExpertWorkflow(grantedKeys: ExpertCredentialRef[]) {
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
    getListExpertCredentialsMockHandler(grantedKeys),
    http.get("*/api/integrations/providers", () =>
      HttpResponse.json([{ name: "firecrawl", description: "firecrawl" }]),
    ),
    http.get("*/api/integrations/providers/system", () =>
      HttpResponse.json(["firecrawl"]),
    ),
    http.get("*/api/integrations/credentials", () =>
      HttpResponse.json([PLATFORM_KEY, USER_KEY]),
    ),
  );
}

function renderScheduleDialog() {
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
}

async function clickWhenEnabled(name: string | RegExp) {
  const button = await screen.findByRole("button", { name });
  await waitFor(() => expect(button.hasAttribute("disabled")).toBe(false));
  await userEvent.click(button);
}

async function openRunForm() {
  // The row is re-rendered as the run form's trigger once the workflow loads
  await waitFor(() =>
    expect(
      screen
        .getByRole("button", { name: /^Scrape agent/ })
        .hasAttribute("disabled"),
    ).toBe(false),
  );
  await userEvent.click(screen.getByRole("button", { name: /^Scrape agent/ }));
}

test("a schedule made for an expert's workflow starts on the platform key, not a step's key the expert was never granted", async () => {
  const scheduleBodies: Array<{ credentials?: unknown }> = [];
  // The expert was granted none of the user's credentials
  mockExpertWorkflow([]);
  server.use(
    getPostV1CreateExecutionScheduleMockHandler(async ({ request }) => {
      scheduleBodies.push((await request.json()) as never);
      return getPostV1CreateExecutionScheduleResponseMock({
        id: "new-schedule",
        graph_id: "graph-1",
      });
    }),
  );
  renderScheduleDialog();

  await openRunForm();
  await clickWhenEnabled("Schedule Task");
  await clickWhenEnabled("Schedule");

  await waitFor(() => expect(scheduleBodies).toHaveLength(1));
  // The backend attributes this schedule to the expert, and refuses every
  // run that names a credential the expert wasn't granted. The platform's
  // key needs no grant.
  expect(keyIn(scheduleBodies[0].credentials)).toBe(PLATFORM_KEY.id);
});

test("a schedule made on the expert page for a step that refuses the platform's key is made on the user's key the run form shows", async () => {
  const accepted: Array<{ credentials?: unknown }> = [];
  const refused: Array<{ credentials?: unknown }> = [];
  // The expert was granted the user's key, so a schedule on it would run
  mockExpertWorkflow([
    {
      credential_id: USER_KEY.id,
      provider: "firecrawl",
      title: USER_KEY.title,
      type: "api_key",
    },
  ]);
  // The backend checks a schedule's credentials when the schedule is made,
  // and a block that keeps what it creates in the vendor's account (Exa
  // Websets, SmartLead campaigns) refuses the platform's key there
  server.use(
    http.post("*/api/graphs/:graphId/schedules", async ({ request }) => {
      const body = (await request.json()) as { credentials?: unknown };
      if (keyIn(body.credentials) === PLATFORM_KEY.id) {
        refused.push(body);
        return HttpResponse.json({ detail: REFUSED }, { status: 400 });
      }
      accepted.push(body);
      return HttpResponse.json(
        getPostV1CreateExecutionScheduleResponseMock({
          id: "new-schedule",
          graph_id: "graph-1",
        }),
      );
    }),
  );
  renderScheduleDialog();

  await openRunForm();
  await waitFor(() =>
    expect(
      (screen.getByRole("combobox", { name: PICKER }) as HTMLSelectElement)
        .value,
    ).toBe(USER_KEY.id),
  );
  await clickWhenEnabled("Schedule Task");
  await clickWhenEnabled("Use the keys shown in the run form");
  await clickWhenEnabled("Schedule");

  await waitFor(() => expect(accepted.length + refused.length).toBe(1));
  expect(refused.map((body) => keyIn(body.credentials))).toEqual([]);
  expect(accepted.map((body) => keyIn(body.credentials))).toEqual([
    USER_KEY.id,
  ]);
});
