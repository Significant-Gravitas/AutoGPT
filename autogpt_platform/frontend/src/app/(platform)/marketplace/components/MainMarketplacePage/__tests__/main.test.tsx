import {
  getGetV2ListStoreAgentsResponseMock,
  getGetV2ListStoreCreatorsResponseMock,
} from "@/app/api/__generated__/endpoints/store/store.msw";
import { render, screen } from "@/tests/integrations/test-utils";
import { MainMarkeplacePage } from "../MainMarketplacePage";
import { beforeEach, describe, expect, test, vi } from "vitest";

const mockUseMainMarketplacePage = vi.hoisted(() => vi.fn());
const flagState = vi.hoisted(() => ({ hireExperts: false }));

vi.mock("../useMainMarketplacePage", () => ({
  useMainMarketplacePage: mockUseMainMarketplacePage,
}));

vi.mock("@/services/feature-flags/use-get-flag", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("@/services/feature-flags/use-get-flag")
    >();
  return {
    ...actual,
    useGetFlag: (flag: string) =>
      flag === actual.Flag.HIRE_EXPERTS
        ? flagState.hireExperts
        : actual.useGetFlag(flag as never),
  };
});

// Sentinel: ExpertsSection renders null for signed-out users, so asserting on
// its copy would pass even if the page's flag gate were deleted. Mocking it as
// an always-visible marker makes mounted-vs-not the thing under test.
vi.mock("../../ExpertsSection/ExpertsSection", () => ({
  ExpertsSection: () => <div data-testid="experts-section-sentinel" />,
}));

describe("MainMarketplacePage", () => {
  beforeEach(() => {
    flagState.hireExperts = false;
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

    expect(screen.getByText(/hand-picked/i)).toBeDefined();
    expect(screen.getByText("Featured Agent")).toBeDefined();
    expect(screen.getByText("All AI Workflows")).toBeDefined();
    expect(screen.getAllByText("Top Agent").length).toBeGreaterThan(0);
    expect(screen.getByText("Creator One")).toBeDefined();
    expect(
      screen.getByRole("button", { name: "Become a Creator" }),
    ).toBeDefined();
  });

  test("flag-off: does not mount ExpertsSection and keeps the pre-experts subtitle", () => {
    render(<MainMarkeplacePage />);

    expect(screen.queryByTestId("experts-section-sentinel")).toBeNull();
    expect(
      screen.getByText("Ready-made automations from the community."),
    ).toBeDefined();
    expect(
      screen.queryByText("Install one on an Expert, or run it standalone."),
    ).toBeNull();
  });

  test("flag-on: mounts ExpertsSection and swaps the workflows subtitle", () => {
    flagState.hireExperts = true;
    render(<MainMarkeplacePage />);

    expect(screen.getByTestId("experts-section-sentinel")).toBeDefined();
    expect(
      screen.getByText("Install one on an Expert, or run it standalone."),
    ).toBeDefined();
    expect(
      screen.queryByText("Ready-made automations from the community."),
    ).toBeNull();
  });
});
