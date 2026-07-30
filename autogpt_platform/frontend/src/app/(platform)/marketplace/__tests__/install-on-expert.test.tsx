import {
  getInstallExpertWorkflowMockHandler,
  getListExpertsMockHandler,
} from "@/app/api/__generated__/endpoints/experts/experts.msw";
import {
  getGetV2GetSpecificAgentMockHandler,
  getGetV2GetSpecificAgentResponseMock,
  getGetV2ListStoreAgentsMockHandler,
  getGetV2ListStoreAgentsResponseMock,
} from "@/app/api/__generated__/endpoints/store/store.msw";
import { Expert } from "@/app/api/__generated__/models/expert";
import { Toaster } from "@/components/molecules/Toast/toaster";
import { server } from "@/mocks/mock-server";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, test, vi } from "vitest";
import { MainAgentPage } from "../components/MainAgentPage/MainAgentPage";

const mockUseAuth = vi.hoisted(() => vi.fn());

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: mockUseAuth,
}));

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return {
    ...actual,
    useGetFlag: (flag: string) =>
      flag === "hire-experts" ? true : actual.useGetFlag(flag as never),
  };
});

const hiredMaria: Expert = {
  id: "expert-maria",
  name: "Maria",
  avatar_url: null,
  role: "Marketing Strategist",
  bio: null,
  skills: [],
  tagline: "Grows your brand while you sleep",
  identity: "You are Maria, a senior marketing strategist.",
  is_template: false,
  source_template_id: "template-maria",
  is_archived: false,
  workflows: [],
};

const agentDetails = getGetV2GetSpecificAgentResponseMock({
  agent_name: "Deterministic Agent",
  creator: "AutoGPT",
  creator_avatar: "",
  sub_heading: "A stable marketplace listing",
  description: "This agent is used for integration coverage.",
  categories: ["demo"],
  versions: ["1"],
  active_version_id: "store-version-1",
  store_listing_version_id: "listing-1",
  agent_image: ["https://example.com/agent.png"],
  agent_output_demo: "",
  agent_video: "",
});

function useStoreHandlers() {
  server.use(
    getGetV2GetSpecificAgentMockHandler(agentDetails),
    getGetV2ListStoreAgentsMockHandler(() =>
      getGetV2ListStoreAgentsResponseMock({ agents: [] }),
    ),
  );
}

function renderAgentPage() {
  return render(
    <>
      <MainAgentPage
        params={{ creator: "autogpt", slug: "deterministic-agent" }}
      />
      <Toaster />
    </>,
  );
}

describe("Install on Expert from marketplace detail", () => {
  beforeEach(() => {
    mockUseAuth.mockReturnValue({
      user: { id: "user-1" },
      isLoggedIn: true,
    });
  });

  test("shows the action and installs on the selected expert", async () => {
    let installBody: unknown = null;
    let installExpertId: string | null = null;
    useStoreHandlers();
    server.use(
      getListExpertsMockHandler([hiredMaria]),
      getInstallExpertWorkflowMockHandler(async ({ request, params }) => {
        installBody = await request.json();
        installExpertId = params.expertId as string;
        return {
          id: "wf-new",
          store_listing_version_id: "listing-1",
          library_agent_id: null,
          graph_id: null,
          name: "Deterministic Agent",
          description: null,
        };
      }),
    );

    renderAgentPage();

    await screen.findByTestId("agent-title");
    await userEvent.click(
      await screen.findByRole("button", { name: "Install on Expert…" }),
    );
    await userEvent.click(
      await screen.findByRole("button", { name: "Install" }),
    );

    expect(await screen.findByText("Installed on Maria")).toBeDefined();
    expect(installExpertId).toBe("expert-maria");
    expect(installBody).toEqual({ store_listing_version_id: "listing-1" });
  });

  test("toasts success when the workflow is already installed", async () => {
    let installCalled = false;
    useStoreHandlers();
    server.use(
      getListExpertsMockHandler([
        {
          ...hiredMaria,
          workflows: [
            {
              id: "wf-1",
              store_listing_version_id: "listing-1",
              library_agent_id: null,
              graph_id: null,
              name: "Deterministic Agent",
              description: null,
            },
          ],
        },
      ]),
      getInstallExpertWorkflowMockHandler(() => {
        installCalled = true;
        return {
          id: "wf-1",
          store_listing_version_id: "listing-1",
          library_agent_id: null,
          graph_id: null,
          name: "Deterministic Agent",
          description: null,
        };
      }),
    );

    renderAgentPage();

    await screen.findByTestId("agent-title");
    await userEvent.click(
      await screen.findByRole("button", { name: "Install on Expert…" }),
    );
    await userEvent.click(
      await screen.findByRole("button", { name: "Install" }),
    );

    expect(await screen.findByText("Already installed on Maria")).toBeDefined();
    expect(installCalled).toBe(false);
  });

  test("hides the action when the user has no hired experts", async () => {
    let expertsRequested = false;
    useStoreHandlers();
    server.use(
      getListExpertsMockHandler(() => {
        expertsRequested = true;
        return [];
      }),
    );

    renderAgentPage();

    await screen.findByTestId("agent-title");
    await waitFor(() => expect(expertsRequested).toBe(true));
    expect(
      screen.queryByRole("button", { name: "Install on Expert…" }),
    ).toBeNull();
  });

  test("hides the action for signed-out visitors without fetching experts", async () => {
    let expertsRequested = false;
    mockUseAuth.mockReturnValue({ user: null, isLoggedIn: false });
    useStoreHandlers();
    server.use(
      getListExpertsMockHandler(() => {
        expertsRequested = true;
        return [hiredMaria];
      }),
    );

    renderAgentPage();

    await screen.findByTestId("agent-title");
    expect(
      screen.queryByRole("button", { name: "Install on Expert…" }),
    ).toBeNull();
    expect(expertsRequested).toBe(false);
  });
});
