import { getGetV2ListStoreAgentsResponseMock } from "@/app/api/__generated__/endpoints/store/store.msw";
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
      isLoading: false,
      hasError: false,
    });
  });

  test("renders featured agents and all agents, and no creators shelf", () => {
    render(<MainMarkeplacePage />);

    expect(screen.getByText(/hand-picked/i)).toBeDefined();
    expect(screen.getByText("Featured Agent")).toBeDefined();
    expect(screen.getByText("All AI Workflows")).toBeDefined();
    expect(screen.getAllByText("Top Agent").length).toBeGreaterThan(0);
    expect(screen.queryByText("Featured Creators")).toBeNull();
    expect(
      screen.getByRole("button", { name: "Become a Creator" }),
    ).toBeDefined();
  });
});
