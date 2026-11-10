import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { makeDashboard } from "@/app/(platform)/home/__tests__/heldItems";
import { getGetV2ListSessionsMockHandler200 } from "@/app/api/__generated__/endpoints/chat/chat.msw";
import {
  getGetExpertMockHandler,
  getListExpertCredentialsMockHandler200,
  getListExpertIdentitiesMockHandler,
  getListExpertsMockHandler200,
} from "@/app/api/__generated__/endpoints/experts/experts.msw";
import {
  getGetV1ListExecutionSchedulesForAUserMockHandler,
  getListCopilotFollowupSchedulesMockHandler,
} from "@/app/api/__generated__/endpoints/schedules/schedules.msw";
import type { Expert } from "@/app/api/__generated__/models/expert";
import { InsetSideSlot } from "@/components/layout/InsetSidePanel/InsetSidePanel";
import { server } from "@/mocks/mock-server";
import { render, screen, within } from "@/tests/integrations/test-utils";
import { withNuqsTestingAdapter } from "nuqs/adapters/testing";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { EmptySession } from "../EmptySession";
import { makeTask } from "./taskFixtures";

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({ user: null, isUserLoading: false, isLoggedIn: true }),
}));

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return {
    ...actual,
    useGetFlag: () => false,
    useFlagStatus: () => ({ enabled: false, ready: true }),
  };
});

const maria: Expert = {
  id: "expert-maria",
  name: "Maria",
  avatar_url: "https://example.com/maria.png",
  role: "Marketing Strategist",
  job_title: "Head of Growth",
  tagline: "Turns launches into leads.",
  bio: null,
  skills: ["Copywriting", "SEO"],
  identity: "You are Maria.",
  voice_preferences: "",
  boundaries: "",
  protected_soul_rules: [],
  is_template: false,
  source_template_id: null,
  is_archived: false,
  workflows: [
    {
      id: "wf-1",
      store_listing_version_id: null,
      library_agent_id: null,
      graph_id: "graph-1",
      name: "Weekly newsletter",
      description: "Sends the digest",
      schedule_id: "schedule-1",
    },
  ],
  autopilot_mode: "ask_first",
  weekly_budget: null,
  last_run_at: null,
};

let taskQueries: URLSearchParams[] = [];

function serveTasks() {
  server.use(
    http.get("/api/proxy/api/tasks", ({ request }) => {
      const params = new URL(request.url).searchParams;
      taskQueries.push(params);
      const owner = params.get("owner_expert_id");
      return HttpResponse.json({
        tasks: owner
          ? [makeTask("task-m", "Draft the launch post", "DONE", owner)]
          : [
              makeTask("task-o", "Plan the offsite", "IN_PROGRESS"),
              makeTask("task-m", "Draft the launch post", "DONE", maria.id),
            ],
      });
    }),
  );
}

function renderHome(searchParams: string) {
  const Wrapper = withNuqsTestingAdapter({ searchParams, hasMemory: true });
  return render(
    <Wrapper>
      <div className="flex">
        <EmptySession
          isCreatingSession={false}
          onCreateSession={() => {}}
          onSend={vi.fn()}
        />
        <InsetSideSlot />
      </div>
    </Wrapper>,
  );
}

async function openPanel(name: string) {
  await userEvent.click(
    await screen.findByRole("button", { name: "Show expert info" }),
  );
  return screen.findByRole("complementary", { name: `About ${name}` });
}

function group(panel: HTMLElement, name: string) {
  return within(panel).findByRole("region", { name });
}

beforeEach(() => {
  taskQueries = [];
  serveTasks();
  server.use(
    getListExpertIdentitiesMockHandler([maria]),
    getListExpertsMockHandler200([maria]),
    getGetExpertMockHandler(maria),
    getListExpertCredentialsMockHandler200([
      {
        credential_id: "cred-1",
        provider: "google",
        title: "Work Gmail",
        type: "oauth2",
        service_name: "Gmail",
      },
    ]),
    getGetV1ListExecutionSchedulesForAUserMockHandler([
      {
        id: "schedule-1",
        user_id: "user-1",
        graph_id: "graph-1",
        graph_version: 1,
        name: "Weekly newsletter run",
        cron: "0 9 * * 1",
        input_data: {},
        expert_id: maria.id,
        next_run_time: "2099-01-05T09:00:00Z",
      },
    ]),
    getListCopilotFollowupSchedulesMockHandler([]),
    http.get(/\/api\/proxy\/api\/home(?:\?.*)?$/, () =>
      HttpResponse.json(makeDashboard([])),
    ),
    getGetV2ListSessionsMockHandler200({ sessions: [], total: 0 }),
  );
});

describe("Home — expert info panel", () => {
  it("shows Otto's profile with what is scheduled and the team's recent tasks", async () => {
    renderHome("");
    const panel = await openPanel("Otto");

    expect(within(panel).getByRole("heading", { name: "Otto" })).toBeDefined();
    expect(within(panel).getByText("Head of AI")).toBeDefined();
    const scheduled = await group(panel, "Scheduled");
    expect(
      await within(scheduled).findByText("Nothing scheduled yet."),
    ).toBeDefined();

    const tasks = await group(panel, "Recent tasks");
    const row = await within(tasks).findByRole("link", {
      name: /Plan the offsite/,
    });
    expect(row.getAttribute("href")).toBe("/tasks?task=task-o");
    expect(row.textContent).toContain("Otto");
    expect(row.textContent).toContain("Working");
    expect(
      within(tasks).getByRole("link", { name: /Draft the launch post/ })
        .textContent,
    ).toContain("Maria");
    expect(taskQueries.at(-1)?.get("owner_expert_id")).toBeNull();
  });

  it("shows an expert's details, access and tasks under their tabs", async () => {
    const user = userEvent.setup();
    renderHome(`?expertId=${maria.id}`);
    const panel = await openPanel("Maria");

    expect(
      await within(panel).findByRole("heading", { name: "Maria" }),
    ).toBeDefined();
    expect(within(panel).getByText("Head of Growth")).toBeDefined();
    expect(within(panel).getByText("Turns launches into leads.")).toBeDefined();

    const scheduled = await group(panel, "Scheduled");
    expect(
      await within(scheduled).findByText("Weekly newsletter run"),
    ).toBeDefined();
    const workInfo = await group(panel, "Work info");
    expect(within(workInfo).getByText("Weekly budget")).toBeDefined();
    expect(within(workInfo).getByText("No limit")).toBeDefined();
    expect(within(workInfo).getByText("Not yet")).toBeDefined();

    await user.click(within(panel).getByRole("radio", { name: "Access" }));
    const connections = await group(panel, "Connections");
    expect(await within(connections).findByText("Gmail")).toBeDefined();
    expect(within(connections).getByText("Work Gmail")).toBeDefined();
    const skills = await group(panel, "Skills");
    expect(within(skills).getByText("Copywriting")).toBeDefined();
    const workflows = await group(panel, "Workflows");
    expect(within(workflows).getByText("Weekly newsletter")).toBeDefined();
    expect(within(workflows).getByText("Scheduled")).toBeDefined();

    await user.click(within(panel).getByRole("radio", { name: "Tasks" }));
    const tasks = await group(panel, "Tasks");
    const row = await within(tasks).findByRole("link", {
      name: /Draft the launch post/,
    });
    expect(row.getAttribute("href")).toBe("/tasks?task=task-m");
    expect(row.textContent).toContain("Done");
    expect(taskQueries.at(-1)?.get("owner_expert_id")).toBe(maria.id);
  });
});
