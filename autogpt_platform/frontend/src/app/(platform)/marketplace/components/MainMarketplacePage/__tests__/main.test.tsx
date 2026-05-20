import {
  getGetV2ListStoreAgentsResponseMock,
  getGetV2ListStoreCreatorsResponseMock,
} from "@/app/api/__generated__/endpoints/store/store.msw";
import { render, screen } from "@/tests/integrations/test-utils";
import { MainMarkeplacePage } from "../MainMarketplacePage";
import { beforeEach, describe, expect, test, vi } from "vitest";

const mockUseMainMarketplacePage = vi.hoisted(() => vi.fn());

vi.mock("../useMainMarketplacePage", () => ({
  useMainMarketplacePage: mockUseMainMarketplacePage,
}));

describe("MainMarketplacePage", () => {
  beforeEach(() => {
    mockUseMainMarketplacePage.mockReturnValue({
      featuredAgents: getGetV2ListStoreAgentsResponseMock({
        agents: [
          {
            ...getGetV2ListStoreAgentsResponseMock().agents[0],
            slug: "featured-agent",
            agent_name: "Featured Agent",
            creator: "AutoGPT",
          },
        ],
      }),
      topAgents: getGetV2ListStoreAgentsResponseMock({
        agents: [
          {
            ...getGetV2ListStoreAgentsResponseMock().agents[0],
            slug: "top-agent",
            agent_name: "Top Agent",
            creator: "AutoGPT",
          },
        ],
      }),
      featuredCreators: getGetV2ListStoreCreatorsResponseMock({
        creators: [
          {
            ...getGetV2ListStoreCreatorsResponseMock().creators[0],
            name: "Creator One",
            username: "creator-one",
          },
        ],
      }),
      isLoading: false,
      hasError: false,
    });
  });

  test("renders featured agents, all agents, and creators", () => {
    render(<MainMarkeplacePage />);

    expect(screen.getByText(/Featured agents/i)).toBeDefined();
    expect(screen.getByText("Featured Agent")).toBeDefined();
    expect(screen.getByText("All Agents")).toBeDefined();
    expect(screen.getAllByText("Top Agent").length).toBeGreaterThan(0);
    expect(screen.getByText("Creator One")).toBeDefined();
    expect(
      screen.getByRole("button", { name: "Become a Creator" }),
    ).toBeDefined();
  });
});
