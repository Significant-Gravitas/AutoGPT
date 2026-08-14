import { getCreateRaisedExpertMockHandler } from "@/app/api/__generated__/endpoints/experts/experts.msw";
import {
  getGetV2GetSpecificAgentMockHandler,
  getGetV2ListStoreAgentsMockHandler,
} from "@/app/api/__generated__/endpoints/store/store.msw";
import type { Expert } from "@/app/api/__generated__/models/expert";
import type { RaiseResult } from "@/app/api/__generated__/models/raiseResult";
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

const { pushMock, notFoundMock } = vi.hoisted(() => ({
  pushMock: vi.fn(),
  notFoundMock: vi.fn(),
}));

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: pushMock }),
  usePathname: () => "/raise",
  useSearchParams: () => new URLSearchParams(),
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
  setFlagStatusMock.mockReturnValue({ enabled: true, ready: true });
  pushMock.mockClear();
  notFoundMock.mockClear();
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

test("shows a friendly limit message on 409", async () => {
  useStoreHandlers();
  server.use(
    http.post("/api/proxy/api/experts/raise", () =>
      HttpResponse.json(
        { detail: "Active expert limit of 20 reached" },
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
