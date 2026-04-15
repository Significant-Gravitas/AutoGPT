import { render, screen } from "@/tests/integrations/test-utils";
import {
  getGetV2GetCreatorDetailsResponseMock,
  getGetV2ListStoreAgentsResponseMock,
} from "@/app/api/__generated__/endpoints/store/store.msw";
import { MainCreatorPage } from "../MainCreatorPage";
import { beforeEach, describe, expect, test, vi } from "vitest";

const mockUseMainCreatorPage = vi.hoisted(() => vi.fn());

vi.mock("../useMainCreatorPage", () => ({
  useMainCreatorPage: mockUseMainCreatorPage,
}));

describe("MainCreatorPage", () => {
  beforeEach(() => {
    const creator = getGetV2GetCreatorDetailsResponseMock({
      name: "Creator One",
      username: "creator-one",
      description: "Creator profile used for integration coverage.",
      avatar_url: "",
      top_categories: ["automation", "productivity"],
      links: ["https://example.com/creator"],
    });

    const creatorAgents = getGetV2ListStoreAgentsResponseMock({
      agents: [
        {
          ...getGetV2ListStoreAgentsResponseMock().agents[0],
          slug: "creator-agent",
          agent_name: "Creator Agent",
          creator: "Creator One",
        },
      ],
    });

    mockUseMainCreatorPage.mockReturnValue({
      creatorAgents,
      creator,
      isLoading: false,
      hasError: false,
    });
  });

  test("renders creator details and their agents", () => {
    render(<MainCreatorPage params={{ creator: "creator-one" }} />);

    expect(screen.getByTestId("creator-title").textContent).toContain(
      "Creator One",
    );
    expect(screen.getByTestId("creator-description").textContent).toContain(
      "Creator profile used for integration coverage.",
    );
    expect(screen.getByText("Agents by Creator One")).toBeDefined();
    expect(screen.getAllByText("Creator Agent").length).toBeGreaterThan(0);
  });
});
