import { getGetV2ListStoreAgentsResponseMock } from "@/app/api/__generated__/endpoints/store/store.msw";
import { Flag } from "@/services/feature-flags/use-get-flag";
import {
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { MainMarkeplacePage } from "../MainMarketplacePage";
import { beforeEach, describe, expect, test, vi } from "vitest";

const mockUseMainMarketplacePage = vi.hoisted(() => vi.fn());
const mockUseFlagSource = vi.hoisted(() => vi.fn());
const onboarding = vi.hoisted(() => ({
  ready: false,
  completeStep: vi.fn(),
}));

vi.mock("@/lib/auth/hooks/useAuth", () => ({
  useAuth: () => ({
    user: { id: "marketplace-intro-user" },
    isLoggedIn: true,
    isUserLoading: false,
  }),
}));

vi.mock("@/providers/onboarding/onboarding-provider", () => ({
  useOnboarding: () => ({
    state: onboarding.ready ? { completedSteps: [] } : null,
    completeStep: onboarding.completeStep,
  }),
  default: ({ children }: { children: React.ReactNode }) => children,
}));

vi.mock("../useMainMarketplacePage", () => ({
  useMainMarketplacePage: mockUseMainMarketplacePage,
}));

vi.mock("@/services/feature-flags/flag-source", () => ({
  useFlagSource: mockUseFlagSource,
}));

describe("MainMarketplacePage", () => {
  beforeEach(() => {
    window.localStorage.clear();
    onboarding.ready = false;
    onboarding.completeStep.mockReset();
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

  test.each([
    { experts: true, emptyShelf: false, trigger: "Publish your workflows" },
    { experts: false, emptyShelf: false, trigger: "Become a Creator" },
    { experts: true, emptyShelf: true, trigger: "Become a Creator" },
  ])(
    "defers the intro while publishing from $trigger (experts=$experts, emptyShelf=$emptyShelf)",
    async ({ experts, emptyShelf, trigger }) => {
      mockUseFlagSource.mockImplementation((flag: string) => ({
        value: flag === Flag.HIRE_EXPERTS ? experts : true,
        resolved: true,
      }));
      if (emptyShelf) {
        const data = mockUseMainMarketplacePage();
        mockUseMainMarketplacePage.mockReturnValue({
          ...data,
          featuredAgents: { ...data.featuredAgents, agents: [] },
          topAgents: { ...data.topAgents, agents: [] },
        });
      }
      const { rerender } = render(<MainMarkeplacePage />);
      await userEvent.click(screen.getByRole("button", { name: trigger }));
      const publishDialog = await screen.findByRole("dialog", {
        name: "Publish Workflow",
      });
      expect(screen.getByTestId("publish-agent-modal")).toBeDefined();

      onboarding.ready = true;
      rerender(<MainMarkeplacePage />);
      expect(screen.queryByTestId("tab-intro-overlay")).toBeNull();
      expect(onboarding.completeStep).not.toHaveBeenCalled();

      mockUseFlagSource.mockImplementation((flag: string) => ({
        value: flag === Flag.HIRE_EXPERTS ? !experts : true,
        resolved: true,
      }));
      rerender(<MainMarkeplacePage />);
      expect(screen.getByRole("dialog", { name: "Publish Workflow" })).toBe(
        publishDialog,
      );
      expect(screen.queryByTestId("tab-intro-overlay")).toBeNull();
      expect(onboarding.completeStep).not.toHaveBeenCalled();

      await userEvent.click(
        within(publishDialog).getByRole("button", {
          name: "Close",
        }),
      );
      expect(
        await screen.findByRole("dialog", { name: "Workflows ready to work." }),
      ).toBeDefined();
      expect(onboarding.completeStep).not.toHaveBeenCalled();
      await userEvent.click(screen.getByRole("button", { name: "Got it" }));
      await waitFor(() =>
        expect(screen.queryByTestId("tab-intro-overlay")).toBeNull(),
      );
      expect(onboarding.completeStep).toHaveBeenCalledWith(
        "MARKETPLACE_TAB_INTRO",
      );
    },
  );
});
