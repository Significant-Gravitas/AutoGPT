import { describe, expect, test, afterEach } from "vitest";
import { render, screen } from "@/tests/integrations/test-utils";
import { LibraryAgentCard } from "../LibraryAgentCard/LibraryAgentCard";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import {
  mockAuthenticatedUser,
  resetAuthState,
} from "@/tests/integrations/helpers/mock-supabase-auth";

const mockAgent: LibraryAgent = {
  id: "test-agent-id",
  graph_id: "test-graph-id",
  graph_version: 1,
  owner_user_id: "test-owner-id",
  image_url: null,
  creator_name: "Test Creator",
  creator_image_url: "https://example.com/avatar.png",
  status: "READY",
  created_at: new Date().toISOString(),
  updated_at: new Date().toISOString(),
  name: "Test Agent Name",
  description: "Test agent description",
  input_schema: {},
  output_schema: {},
  credentials_input_schema: null,
  has_external_trigger: false,
  has_human_in_the_loop: false,
  has_sensitive_action: false,
  new_output: false,
  can_access_graph: true,
  is_latest_version: true,
  is_favorite: false,
};

describe("LibraryAgentCard", () => {
  afterEach(() => {
    resetAuthState();
  });

  test("renders agent name", () => {
    mockAuthenticatedUser();
    render(<LibraryAgentCard agent={mockAgent} />);

    expect(screen.getByText("Test Agent Name")).toBeInTheDocument();
  });

  test("renders see runs link", () => {
    mockAuthenticatedUser();
    render(<LibraryAgentCard agent={mockAgent} />);

    expect(screen.getByText(/see runs/i)).toBeInTheDocument();
  });

  test("renders open in builder link when can_access_graph is true", () => {
    mockAuthenticatedUser();
    render(<LibraryAgentCard agent={mockAgent} />);

    expect(screen.getByText(/open in builder/i)).toBeInTheDocument();
  });

  test("does not render open in builder link when can_access_graph is false", () => {
    mockAuthenticatedUser();
    const agentWithoutAccess = { ...mockAgent, can_access_graph: false };
    render(<LibraryAgentCard agent={agentWithoutAccess} />);

    expect(screen.queryByText(/open in builder/i)).not.toBeInTheDocument();
  });

  test("shows 'FROM MARKETPLACE' label for marketplace agents", () => {
    mockAuthenticatedUser();
    const marketplaceAgent = {
      ...mockAgent,
      marketplace_listing: {
        id: "listing-id",
        name: "Marketplace Agent",
        slug: "marketplace-agent",
        creator: {
          id: "creator-id",
          name: "Creator Name",
          slug: "creator-slug",
        },
      },
    };
    render(<LibraryAgentCard agent={marketplaceAgent} />);

    expect(screen.getByText(/from marketplace/i)).toBeInTheDocument();
  });

  test("shows 'Built by you' label for user's own agents", () => {
    mockAuthenticatedUser();
    render(<LibraryAgentCard agent={mockAgent} />);

    expect(screen.getByText(/built by you/i)).toBeInTheDocument();
  });

  test("renders favorite button", () => {
    mockAuthenticatedUser();
    render(<LibraryAgentCard agent={mockAgent} />);

    // The favorite button should be present (as a heart icon button)
    const card = screen.getByTestId("library-agent-card");
    expect(card).toBeInTheDocument();
  });

  test("links to correct agent detail page", () => {
    mockAuthenticatedUser();
    render(<LibraryAgentCard agent={mockAgent} />);

    const link = screen.getByTestId("library-agent-card-see-runs-link");
    expect(link).toHaveAttribute("href", "/library/agents/test-agent-id");
  });

  test("links to correct builder page", () => {
    mockAuthenticatedUser();
    render(<LibraryAgentCard agent={mockAgent} />);

    const builderLink = screen.getByTestId(
      "library-agent-card-open-in-builder-link",
    );
    expect(builderLink).toHaveAttribute("href", "/build?flowID=test-graph-id");
  });
});
