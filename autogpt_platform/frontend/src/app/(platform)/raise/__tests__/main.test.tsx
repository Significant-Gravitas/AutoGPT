import { getCreateRaisedExpertMockHandler } from "@/app/api/__generated__/endpoints/experts/experts.msw";
import {
  getGetV2GetSpecificAgentMockHandler,
  getGetV2ListStoreAgentsMockHandler,
} from "@/app/api/__generated__/endpoints/store/store.msw";
import type { Expert } from "@/app/api/__generated__/models/expert";
import type { StoreAgent } from "@/app/api/__generated__/models/storeAgent";
import type { StoreAgentDetails } from "@/app/api/__generated__/models/storeAgentDetails";
import type { StoreAgentsResponse } from "@/app/api/__generated__/models/storeAgentsResponse";
import { Toaster } from "@/components/molecules/Toast/toaster";
import { server } from "@/mocks/mock-server";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
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

beforeEach(() => {
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

test("walks name → voice → first job and posts the assembled payload", async () => {
  let captured: unknown = null;
  useStoreHandlers();
  server.use(
    getCreateRaisedExpertMockHandler(async (info) => {
      captured = await info.request.json();
      return raisedExpert;
    }),
  );

  renderRaise();

  await userEvent.click(await screen.findByRole("button", { name: "Otto" }));

  await userEvent.click(await screen.findByText("Concise and direct"));
  await userEvent.click(screen.getByRole("button", { name: "Use this voice" }));

  await userEvent.click(await screen.findByText("SEO Blog Writer"));
  const confirmJob = (await screen.findByRole("button", {
    name: "Give me this job",
  })) as HTMLButtonElement;
  await waitFor(() => expect(confirmJob.disabled).toBe(false));
  await userEvent.click(confirmJob);

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

test("skips voice and first job and posts a minimal payload", async () => {
  let captured: unknown = null;
  useStoreHandlers();
  server.use(
    getCreateRaisedExpertMockHandler(async (info) => {
      captured = await info.request.json();
      return raisedExpert;
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
  await waitFor(() => expect(pushMock).toHaveBeenCalled());
});
