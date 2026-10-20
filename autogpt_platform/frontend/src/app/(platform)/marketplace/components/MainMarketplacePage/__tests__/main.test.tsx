import { getGetV2ListStoreAgentsResponseMock } from "@/app/api/__generated__/endpoints/store/store.msw";
import { Flag } from "@/services/feature-flags/use-get-flag";
import { render, screen } from "@/tests/integrations/test-utils";
import { MainMarkeplacePage } from "../MainMarketplacePage";
import { beforeEach, describe, expect, test, vi } from "vitest";

const mockUseMainMarketplacePage = vi.hoisted(() => vi.fn());
const mockUseFlagSource = vi.hoisted(() => vi.fn());

vi.mock("../useMainMarketplacePage", () => ({
  useMainMarketplacePage: mockUseMainMarketplacePage,
}));

vi.mock("@/services/feature-flags/flag-source", () => ({
  useFlagSource: mockUseFlagSource,
}));

describe("MainMarketplacePage", () => {
  beforeEach(() => {
    mockUseFlagSource.mockReturnValue({
      value: undefined,
      resolved: false,
    });
    mockUseMainMarketplacePage.mockReturnValue({
      featuredAgents: getGetV2ListStoreAgentsResponseMock({
        agents: [
          {
            ...getGetV2ListStoreAgentsResponseMock().agents[0],
            agent_graph_id: "featured-graph",
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
            agent_graph_id: "top-graph",
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

  test("renders the expert layout and workflow shelf before flags answer", () => {
    render(<MainMarkeplacePage />);

    expect(
      screen.getByRole("heading", { name: /hire an AI expert/i }),
    ).toBeDefined();
    expect(screen.getByRole("region", { name: "Workflows" })).toBeDefined();
    expect(screen.getByText("Featured Agent")).toBeDefined();
    expect(screen.getByText("Top Agent")).toBeDefined();
    expect(
      screen.getByRole("button", { name: "Publish your workflows" }),
    ).toBeDefined();
    expect(screen.queryByText(/hand-picked/i)).toBeNull();
    expect(screen.queryByText("All AI Workflows")).toBeNull();
    expect(screen.queryByText("Featured Creators")).toBeNull();
  });

  test("renders the legacy workflow catalogue when hire-experts is explicitly off", () => {
    mockUseFlagSource.mockImplementation((flag: string) => ({
      value: flag === Flag.HIRE_EXPERTS ? false : undefined,
      resolved: flag === Flag.HIRE_EXPERTS,
    }));

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
