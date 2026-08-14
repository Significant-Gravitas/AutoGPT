import { NamingMomentCard } from "@/app/(platform)/copilot/components/NamingMomentCard/NamingMomentCard";
import { getGetV2ListSessionsMockHandler } from "@/app/api/__generated__/endpoints/chat/chat.msw";
import { useListExperts } from "@/app/api/__generated__/endpoints/experts/experts";
import {
  getCreateRaisedExpertMockHandler,
  getListExpertsMockHandler,
} from "@/app/api/__generated__/endpoints/experts/experts.msw";
import {
  getGetV2GetSpecificAgentMockHandler,
  getGetV2ListStoreAgentsMockHandler,
} from "@/app/api/__generated__/endpoints/store/store.msw";
import type { Expert } from "@/app/api/__generated__/models/expert";
import type { RaiseResult } from "@/app/api/__generated__/models/raiseResult";
import type { SessionSummaryResponse } from "@/app/api/__generated__/models/sessionSummaryResponse";
import type { StoreAgent } from "@/app/api/__generated__/models/storeAgent";
import type { StoreAgentDetails } from "@/app/api/__generated__/models/storeAgentDetails";
import type { StoreAgentsResponse } from "@/app/api/__generated__/models/storeAgentsResponse";
import { Toaster } from "@/components/molecules/Toast/toaster";
import { server } from "@/mocks/mock-server";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { HttpResponse, http } from "msw";
import { afterEach, beforeEach, expect, test, vi } from "vitest";
import RaisePage from "../page";

const { setFlagStatusMock } = vi.hoisted(() => ({
  setFlagStatusMock: vi.fn(() => ({ enabled: true, ready: true })),
}));

vi.mock("@/services/feature-flags/use-get-flag", async (importActual) => {
  const actual =
    await importActual<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return {
    ...actual,
    useFlagStatus: (flag: string) =>
      flag === "hire-experts"
        ? setFlagStatusMock()
        : actual.useFlagStatus(flag as never),
  };
});

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({ isLoggedIn: true, user: { id: "user-1" } }),
}));

const { pushMock, notFoundMock, searchParamsMock } = vi.hoisted(() => ({
  pushMock: vi.fn(),
  notFoundMock: vi.fn(),
  searchParamsMock: { current: new URLSearchParams() },
}));

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: pushMock }),
  usePathname: () => "/raise",
  useSearchParams: () => searchParamsMock.current,
  useParams: () => ({}),
  notFound: () => {
    notFoundMock();
    throw new Error("NEXT_NOT_FOUND");
  },
}));

const raisedExpert = {
  id: "raised-1",
  name: "Otto",
  avatar_url: null,
  role: "",
  tagline: null,
  bio: null,
  skills: [],
  identity: "I'm Otto, raised by you. I learn how you work and grow with you.",
  voice_preferences: "",
  boundaries: "",
  protected_soul_rules: [],
  is_template: false,
  source_template_id: null,
  is_archived: false,
  workflows: [],
} as Expert;

const installedWorkflow = {
  id: "workflow-1",
  store_listing_version_id: "listing-version-42",
  library_agent_id: "library-agent-1",
  graph_id: "graph-1",
  name: "SEO Blog Writer",
  description: null,
};

const storeAgent = {
  slug: "seo-writer",
  agent_name: "SEO Blog Writer",
  agent_image: "",
  creator: "acme",
  creator_avatar: "",
  sub_heading: "Writes optimized blog posts",
  description: "",
  runs: 1200,
  rating: 4.8,
  agent_graph_id: "graph-1",
} as StoreAgent;

const storeAgentsResponse: StoreAgentsResponse = {
  agents: [storeAgent],
  pagination: {
    total_items: 1,
    total_pages: 1,
    current_page: 1,
    page_size: 3,
  },
};

const agentDetails = {
  store_listing_version_id: "listing-version-42",
  slug: "seo-writer",
  agent_name: "SEO Blog Writer",
  creator: "acme",
} as StoreAgentDetails;

function useStoreHandlers() {
  server.use(
    getGetV2ListStoreAgentsMockHandler(storeAgentsResponse),
    getGetV2GetSpecificAgentMockHandler(agentDetails),
  );
}

function renderRaise() {
  return render(
    <>
      <RaisePage />
      <Toaster />
    </>,
  );
}

async function walkToReviewWithJob() {
  await userEvent.click(await screen.findByRole("button", { name: "Otto" }));

  await userEvent.click(await screen.findByText("Concise and direct"));
  await userEvent.click(screen.getByRole("button", { name: "Use this voice" }));

  await userEvent.click(await screen.findByText("SEO Blog Writer"));
  const confirmJob = (await screen.findByRole("button", {
    name: "Give me this job",
  })) as HTMLButtonElement;
  await waitFor(() => expect(confirmJob.disabled).toBe(false));
  await userEvent.click(confirmJob);
}

beforeEach(() => {
  window.sessionStorage.clear();
  window.localStorage.clear();
  setFlagStatusMock.mockReturnValue({ enabled: true, ready: true });
  pushMock.mockClear();
  notFoundMock.mockClear();
  searchParamsMock.current = new URLSearchParams();
});

afterEach(() => {
  vi.clearAllMocks();
});

test("calls notFound when the experts feature is disabled", () => {
  setFlagStatusMock.mockReturnValue({ enabled: false, ready: true });

  try {
    renderRaise();
  } catch {}

  expect(notFoundMock).toHaveBeenCalled();
});

test("walks name → voice → first job, posts the payload, and kicks off", async () => {
  let captured: unknown = null;
  useStoreHandlers();
  server.use(
    getCreateRaisedExpertMockHandler(async (info) => {
      captured = await info.request.json();
      return {
        expert: { ...raisedExpert, workflows: [installedWorkflow] },
        first_job_installed: true,
        first_job_failure_reason: null,
      } as RaiseResult;
    }),
  );

  renderRaise();
  await walkToReviewWithJob();

  await userEvent.click(
    await screen.findByRole("button", { name: /Bring Otto to life/ }),
  );

  await waitFor(() => expect(captured).not.toBeNull());
  expect(captured).toMatchObject({
    name: "Otto",
    first_job_store_listing_version_id: "listing-version-42",
  });
  expect(
    (captured as { voice_preferences: string }).voice_preferences,
  ).toContain("Concise and direct");

  await waitFor(() =>
    expect(pushMock).toHaveBeenCalledWith(
      "/copilot?expertId=raised-1&kickoff=1",
    ),
  );
});

test("skips voice and first job, posts a minimal payload without kickoff", async () => {
  let captured: unknown = null;
  useStoreHandlers();
  server.use(
    getCreateRaisedExpertMockHandler(async (info) => {
      captured = await info.request.json();
      return {
        expert: raisedExpert,
        first_job_installed: false,
        first_job_failure_reason: null,
      } as RaiseResult;
    }),
  );

  renderRaise();

  const nameInput = await screen.findByPlaceholderText("Type a name…");
  await userEvent.type(nameInput, "Juno");
  await userEvent.click(screen.getByRole("button", { name: "Name me" }));

  await userEvent.click(
    await screen.findByRole("button", { name: "Skip for now" }),
  );

  await userEvent.click(
    await screen.findByRole("button", { name: "Skip for now" }),
  );

  await userEvent.click(await screen.findByRole("button", { name: /life/ }));

  await waitFor(() => expect(captured).not.toBeNull());
  expect(
    (captured as Record<string, unknown>).first_job_store_listing_version_id,
  ).toBeNull();
  await waitFor(() =>
    expect(pushMock).toHaveBeenCalledWith("/copilot?expertId=raised-1"),
  );
});

test("surfaces a failed first-job install and skips kickoff", async () => {
  useStoreHandlers();
  server.use(
    getCreateRaisedExpertMockHandler({
      expert: raisedExpert,
      first_job_installed: false,
      first_job_failure_reason: "installation_failed",
    } as RaiseResult),
  );

  renderRaise();
  await walkToReviewWithJob();

  await userEvent.click(
    await screen.findByRole("button", { name: /Bring Otto to life/ }),
  );

  expect(
    await screen.findByText("Couldn't set up Otto's first job"),
  ).toBeDefined();
  expect(screen.getByText(/from their page anytime/)).toBeDefined();
  await waitFor(() =>
    expect(pushMock).toHaveBeenCalledWith("/copilot?expertId=raised-1"),
  );
});

test("a rapid double-click on finish sends a single POST", async () => {
  let postCount = 0;
  useStoreHandlers();
  server.use(
    getCreateRaisedExpertMockHandler(() => {
      postCount += 1;
      return {
        expert: raisedExpert,
        first_job_installed: false,
        first_job_failure_reason: null,
      } as RaiseResult;
    }),
  );

  renderRaise();

  await userEvent.click(await screen.findByRole("button", { name: "Otto" }));
  await userEvent.click(
    await screen.findByRole("button", { name: "Skip for now" }),
  );
  await userEvent.click(
    await screen.findByRole("button", { name: "Skip for now" }),
  );

  const finishButton = await screen.findByRole("button", { name: /life/ });
  await Promise.all([
    userEvent.click(finishButton),
    userEvent.click(finishButton),
  ]);

  await waitFor(() => expect(pushMock).toHaveBeenCalled());
  expect(postCount).toBe(1);
});

test("keeps navigation locked after success until the route unmounts", async () => {
  useStoreHandlers();
  server.use(
    getCreateRaisedExpertMockHandler({
      expert: raisedExpert,
      first_job_installed: false,
      first_job_failure_reason: null,
    } as RaiseResult),
  );

  renderRaise();
  await userEvent.click(await screen.findByRole("button", { name: "Otto" }));
  await userEvent.click(
    await screen.findByRole("button", { name: "Skip for now" }),
  );
  await userEvent.click(
    await screen.findByRole("button", { name: "Skip for now" }),
  );
  await userEvent.click(await screen.findByRole("button", { name: /life/ }));

  await waitFor(() => expect(pushMock).toHaveBeenCalled());
  expect(
    (screen.getByRole("button", { name: "Back" }) as HTMLButtonElement)
      .disabled,
  ).toBe(true);
});

test("shows a friendly limit message on 409", async () => {
  useStoreHandlers();
  server.use(
    http.post("/api/proxy/api/experts/raise", () =>
      HttpResponse.json(
        { detail: { code: "active_expert_limit", limit: 20 } },
        { status: 409 },
      ),
    ),
  );

  renderRaise();

  await userEvent.click(await screen.findByRole("button", { name: "Otto" }));
  await userEvent.click(
    await screen.findByRole("button", { name: "Skip for now" }),
  );
  await userEvent.click(
    await screen.findByRole("button", { name: "Skip for now" }),
  );
  await userEvent.click(await screen.findByRole("button", { name: /life/ }));

  expect(await screen.findByText("Your team is full")).toBeDefined();
  expect(pushMock).not.toHaveBeenCalled();
});

test("distinguishes the lifetime raised-expert limit", async () => {
  useStoreHandlers();
  server.use(
    http.post("/api/proxy/api/experts/raise", () =>
      HttpResponse.json(
        {
          detail: { code: "raised_expert_lifetime_limit", limit: 100 },
        },
        { status: 409 },
      ),
    ),
  );

  renderRaise();
  await userEvent.click(await screen.findByRole("button", { name: "Otto" }));
  await userEvent.click(
    await screen.findByRole("button", { name: "Skip for now" }),
  );
  await userEvent.click(
    await screen.findByRole("button", { name: "Skip for now" }),
  );
  await userEvent.click(await screen.findByRole("button", { name: /life/ }));

  expect(
    await screen.findByText("Expert creation limit reached"),
  ).toBeDefined();
  expect(screen.getByText(/Contact support/)).toBeDefined();
  expect(pushMock).not.toHaveBeenCalled();
});

test("back returns to the previous step and the draft survives", async () => {
  useStoreHandlers();
  renderRaise();

  await userEvent.click(await screen.findByRole("button", { name: "Otto" }));
  expect(await screen.findByText("How should Otto write?")).toBeDefined();

  await userEvent.click(screen.getByRole("button", { name: "Back" }));

  expect(await screen.findByPlaceholderText("Type a name…")).toBeDefined();
  expect(screen.getByText("Otto's Soul")).toBeDefined();
});

test("a refresh resumes the draft from session storage", async () => {
  useStoreHandlers();
  const first = renderRaise();

  await userEvent.click(await screen.findByRole("button", { name: "Nova" }));
  expect(await screen.findByText("How should Nova write?")).toBeDefined();

  first.unmount();
  renderRaise();

  expect(await screen.findByText("How should Nova write?")).toBeDefined();
  expect(screen.getByText("Nova's Soul")).toBeDefined();
});

test("explains when the selected first job becomes unavailable", async () => {
  useStoreHandlers();
  server.use(
    getCreateRaisedExpertMockHandler({
      expert: raisedExpert,
      first_job_installed: false,
      first_job_failure_reason: "unavailable",
    } as RaiseResult),
  );

  renderRaise();
  await walkToReviewWithJob();
  await userEvent.click(
    await screen.findByRole("button", { name: /Bring Otto to life/ }),
  );

  expect(
    await screen.findByText("SEO Blog Writer is no longer available"),
  ).toBeDefined();
  expect(screen.getByText(/choose another first job/)).toBeDefined();
});

test("keeps skip available when starter-job suggestions fail", async () => {
  server.use(
    http.get("/api/proxy/api/store/agents", () =>
      HttpResponse.json({ detail: "Store unavailable" }, { status: 500 }),
    ),
  );

  renderRaise();
  await userEvent.click(await screen.findByRole("button", { name: "Otto" }));
  await userEvent.click(
    await screen.findByRole("button", { name: "Skip for now" }),
  );

  expect(await screen.findByText("Something went wrong")).toBeDefined();
  expect(screen.getByRole("button", { name: "Skip for now" })).toBeDefined();
});

test("explains when there are no starter jobs", async () => {
  server.use(
    getGetV2ListStoreAgentsMockHandler({
      ...storeAgentsResponse,
      agents: [],
      pagination: { ...storeAgentsResponse.pagination, total_items: 0 },
    }),
  );

  renderRaise();
  await userEvent.click(await screen.findByRole("button", { name: "Otto" }));
  await userEvent.click(
    await screen.findByRole("button", { name: "Skip for now" }),
  );

  expect(
    await screen.findByText(/No starter jobs are available right now/),
  ).toBeDefined();
  expect(screen.getByRole("button", { name: "Skip for now" })).toBeDefined();
});

test("surfaces a selected job detail failure", async () => {
  server.use(
    getGetV2ListStoreAgentsMockHandler(storeAgentsResponse),
    http.get("/api/proxy/api/store/agents/:username/:agentName", () =>
      HttpResponse.json({ detail: "Details unavailable" }, { status: 500 }),
    ),
  );

  renderRaise();
  await userEvent.click(await screen.findByRole("button", { name: "Otto" }));
  await userEvent.click(
    await screen.findByRole("button", { name: "Skip for now" }),
  );
  await userEvent.click(await screen.findByText("SEO Blog Writer"));

  expect(await screen.findByText("Something went wrong")).toBeDefined();
  const confirm = screen.getByRole("button", { name: "Give me this job" });
  expect((confirm as HTMLButtonElement).disabled).toBe(true);
  expect(screen.getByRole("button", { name: "Skip for now" })).toBeDefined();
});

test("from=naming shows the naming opener and skips the first job step", async () => {
  searchParamsMock.current = new URLSearchParams("from=naming");
  let captured: unknown = null;
  server.use(
    getCreateRaisedExpertMockHandler(async (info) => {
      captured = await info.request.json();
      return {
        expert: raisedExpert,
        first_job_installed: false,
      } as RaiseResult;
    }),
  );

  renderRaise();

  expect(await screen.findByText("Let's make it official.")).toBeTruthy();

  const nameInput = await screen.findByPlaceholderText("Type a name…");
  await userEvent.type(nameInput, "Juno");
  await userEvent.click(screen.getByRole("button", { name: "Name me" }));
  await userEvent.click(
    await screen.findByRole("button", { name: "Skip for now" }),
  );
  await userEvent.click(await screen.findByRole("button", { name: /life/ }));

  await waitFor(() => expect(captured).not.toBeNull());
  expect(screen.queryByText("SEO Blog Writer")).toBeNull();
  expect(
    (captured as Record<string, unknown>).first_job_store_listing_version_id,
  ).toBeNull();
  await waitFor(() =>
    expect(pushMock).toHaveBeenCalledWith("/copilot?expertId=raised-1"),
  );
});

function ExpertsCacheProbe() {
  const query = useListExperts({
    query: {
      select: (response) =>
        response.status === 200 ? response.data : undefined,
    },
  });
  const names = (query.data ?? []).map((expert) => expert.name);
  return (
    <div>{`cached-experts:${names.length ? names.join(",") : "none"}`}</div>
  );
}

const aSession = {
  id: "session-1",
  created_at: "2026-08-14T00:00:00Z",
  updated_at: "2026-08-14T00:00:00Z",
  is_processing: false,
} as SessionSummaryResponse;

test("finishing naming reconciles a cached empty experts list before navigation", async () => {
  searchParamsMock.current = new URLSearchParams("from=naming");
  let created = false;
  server.use(
    getListExpertsMockHandler(() => (created ? [raisedExpert] : [])),
    getGetV2ListSessionsMockHandler({ sessions: [aSession], total: 2 }),
    getCreateRaisedExpertMockHandler(() => {
      created = true;
      return {
        expert: raisedExpert,
        first_job_installed: false,
      } as RaiseResult;
    }),
  );

  render(
    <>
      <RaisePage />
      <NamingMomentCard />
      <ExpertsCacheProbe />
    </>,
  );

  expect(await screen.findByText("cached-experts:none")).toBeTruthy();
  expect(
    await screen.findByRole("button", { name: "Give me a name" }),
  ).toBeTruthy();

  const nameInput = await screen.findByPlaceholderText("Type a name…");
  await userEvent.type(nameInput, "Otto");
  await userEvent.click(screen.getByRole("button", { name: "Name me" }));
  await userEvent.click(
    await screen.findByRole("button", { name: "Skip for now" }),
  );
  await userEvent.click(await screen.findByRole("button", { name: /life/ }));

  await waitFor(() =>
    expect(pushMock).toHaveBeenCalledWith("/copilot?expertId=raised-1"),
  );
  expect(await screen.findByText("cached-experts:Otto")).toBeTruthy();
  await waitFor(() =>
    expect(screen.queryByRole("button", { name: "Give me a name" })).toBeNull(),
  );
});
