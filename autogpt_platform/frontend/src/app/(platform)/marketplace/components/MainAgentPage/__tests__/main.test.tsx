import { render, screen } from "@/tests/integrations/test-utils";
import {
  getGetV2GetSpecificAgentResponseMock,
  getGetV2ListStoreAgentsResponseMock,
} from "@/app/api/__generated__/endpoints/store/store.msw";
import { MainAgentPage } from "../MainAgentPage";
import { beforeEach, describe, expect, test, vi } from "vitest";

const mockUseMainAgentPage = vi.hoisted(() => vi.fn());

vi.mock("../useMainAgentPage", () => ({
  useMainAgentPage: mockUseMainAgentPage,
}));

describe("MainAgentPage", () => {
  beforeEach(() => {
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

    mockUseMainAgentPage.mockReturnValue({
      agent: { status: 200, data: agentDetails },
      user: null,
      isLoading: false,
      hasError: false,
      similarAgents,
      otherAgents,
      libraryAgent: null,
    });
  });

  test("renders the marketplace agent details and related sections", () => {
    render(
      <MainAgentPage
        params={{ creator: "autogpt", slug: "deterministic-agent" }}
      />,
    );

    expect(screen.getByTestId("agent-title").textContent).toContain(
      "Deterministic Agent",
    );
    expect(screen.getByTestId("agent-description").textContent).toContain(
      "This agent is used for integration coverage.",
    );
    expect(screen.getByTestId("agent-creator").textContent).toContain(
      "AutoGPT",
    );
    expect(screen.getByText("Other agents by AutoGPT")).toBeDefined();
    expect(screen.getByText("Similar agents")).toBeDefined();
  });
});
