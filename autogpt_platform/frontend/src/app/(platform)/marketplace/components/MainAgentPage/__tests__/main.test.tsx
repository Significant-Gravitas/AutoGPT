import {
  getGetV2GetSpecificAgentMockHandler,
  getGetV2GetSpecificAgentResponseMock,
  getGetV2ListStoreAgentsMockHandler,
  getGetV2ListStoreAgentsResponseMock,
} from "@/app/api/__generated__/endpoints/store/store.msw";
import {
  getGetV2GetAgentByStoreIdMockHandler,
  getGetV2GetAgentByStoreIdResponseMock,
  getGetV2ListLibraryAgentsMockHandler,
  getGetV2ListLibraryAgentsResponseMock,
} from "@/app/api/__generated__/endpoints/library/library.msw";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { server } from "@/mocks/mock-server";
import { useOrgTeamStore } from "@/services/org-team/store";
import { render, screen } from "@/tests/integrations/test-utils";
import { MainAgentPage } from "../MainAgentPage";
import { beforeEach, describe, expect, test, vi } from "vitest";

const mockUseAuth = vi.hoisted(() => vi.fn());

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: mockUseAuth,
}));

describe("MainAgentPage", () => {
  beforeEach(() => {
    mockUseAuth.mockReturnValue({
      user: null,
    });
    useOrgTeamStore.setState({
      activeOrgID: null,
      activeTeamID: null,
      orgs: [],
      teams: [],
      isLoaded: false,
    });
  });

  test("renders the marketplace agent details and related sections", async () => {
    const agentDetails = getGetV2GetSpecificAgentResponseMock({
      agent_name: "Deterministic Agent",
      creator: "AutoGPT",
      creator_avatar: "",
      sub_heading: "A stable marketplace listing",
      description: "This agent is used for integration coverage.",
      categories: ["demo", "test"],
      versions: ["1", "2"],
      active_version_id: "store-version-1",
      store_listing_version_id: "listing-1",
      agent_image: ["https://example.com/agent.png"],
      agent_output_demo: "",
      agent_video: "",
    });
    const otherAgents = getGetV2ListStoreAgentsResponseMock({
      agents: [
        {
          ...getGetV2ListStoreAgentsResponseMock().agents[0],
          slug: "other-agent",
          agent_name: "Other Agent",
          creator: "AutoGPT",
        },
      ],
    });
    const similarAgents = getGetV2ListStoreAgentsResponseMock({
      agents: [
        {
          ...getGetV2ListStoreAgentsResponseMock().agents[0],
          slug: "similar-agent",
          agent_name: "Similar Agent",
          creator: "Another Creator",
        },
      ],
    });

    server.use(
      getGetV2GetSpecificAgentMockHandler(agentDetails),
      getGetV2ListStoreAgentsMockHandler(({ request }) => {
        const url = new URL(request.url);

        if (url.searchParams.get("creator") === "autogpt") {
          return otherAgents;
        }

        if (url.searchParams.get("search_query") === "deterministic agent") {
          return similarAgents;
        }

        return getGetV2ListStoreAgentsResponseMock({ agents: [] });
      }),
    );

    render(
      <MainAgentPage
        params={{ creator: "autogpt", slug: "deterministic-agent" }}
      />,
    );

    expect((await screen.findByTestId("agent-title")).textContent).toContain(
      "Deterministic Agent",
    );
    expect(screen.getByTestId("agent-description").textContent).toContain(
      "This agent is used for integration coverage.",
    );
    expect(screen.getByTestId("agent-creator").textContent).toContain(
      "AutoGPT",
    );
    expect(screen.getByText("Other AI workflows by AutoGPT")).toBeDefined();
    expect(screen.getByText("Similar AI workflows")).toBeDefined();
  });

  test("keeps uninstalled tenant targets available when the active team has a copy", async () => {
    mockUseAuth.mockReturnValue({
      user: { id: "user-1" },
      isLoggedIn: true,
      isUserLoading: false,
    });
    useOrgTeamStore.setState({
      activeOrgID: "org-1",
      activeTeamID: "team-a",
      orgs: [],
      teams: [
        {
          id: "team-a",
          name: "Growth",
          slug: "growth",
          isDefault: false,
          joinPolicy: "closed",
          orgId: "org-1",
        },
        {
          id: "team-b",
          name: "Design",
          slug: "design",
          isDefault: false,
          joinPolicy: "closed",
          orgId: "org-1",
        },
      ],
      isLoaded: true,
    });
    const agentDetails = getGetV2GetSpecificAgentResponseMock({
      agent_name: "Target-aware Agent",
      creator: "AutoGPT",
      graph_id: "graph-1",
      active_version_id: "store-version-1",
      store_listing_version_id: "listing-1",
    });
    const libraryAgent = {
      ...getGetV2GetAgentByStoreIdResponseMock()!,
      id: "lib-a",
      name: "Target-aware Agent",
      graph_id: "graph-1",
      organization_id: "org-1",
      team_id: "team-a",
    } satisfies LibraryAgent;
    server.use(
      getGetV2GetSpecificAgentMockHandler(agentDetails),
      getGetV2ListStoreAgentsMockHandler(
        getGetV2ListStoreAgentsResponseMock({ agents: [] }),
      ),
      getGetV2GetAgentByStoreIdMockHandler(libraryAgent),
      getGetV2ListLibraryAgentsMockHandler(
        getGetV2ListLibraryAgentsResponseMock({
          agents: [
            { ...libraryAgent, id: "lib-home", team_id: null },
            libraryAgent,
          ],
        }),
      ),
    );

    render(
      <MainAgentPage
        params={{ creator: "autogpt", slug: "target-aware-agent" }}
      />,
    );

    expect(
      await screen.findByRole("button", { name: "See runs" }),
    ).toBeDefined();
    expect(
      await screen.findByRole("button", {
        name: "Add Target-aware Agent to Design",
      }),
    ).toBeDefined();
  });
});
