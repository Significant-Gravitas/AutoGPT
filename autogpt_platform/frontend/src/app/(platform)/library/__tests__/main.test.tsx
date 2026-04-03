import { describe, expect, test } from "vitest";
import { render, screen } from "@/tests/integrations/test-utils";
import { server } from "@/mocks/mock-server";
import { http, HttpResponse } from "msw";
import LibraryPage from "../page";

const AGENTS_URL = "http://localhost:3000/api/proxy/api/library/agents";
const FAVORITE_AGENTS_URL =
  "http://localhost:3000/api/proxy/api/library/agents/favorites";
const FOLDERS_URL = "http://localhost:3000/api/proxy/api/library/folders";
const EXECUTIONS_URL = "http://localhost:3000/api/proxy/api/executions";

function makeAgent(overrides: Record<string, unknown> = {}) {
  return {
    id: "agent-1",
    graph_id: "graph-1",
    graph_version: 1,
    image_url: null,
    creator_name: "Test Creator",
    creator_image_url: null,
    status: "HEALTHY",
    created_at: "2025-01-01T00:00:00Z",
    updated_at: "2025-01-01T00:00:00Z",
    name: "My Test Agent",
    description: "A test agent",
    instructions: null,
    input_schema: {},
    output_schema: {},
    credentials_input_schema: null,
    has_external_trigger: false,
    has_human_in_the_loop: false,
    has_sensitive_action: false,
    trigger_setup_info: null,
    new_output: false,
    execution_count: 5,
    is_favorite: false,
    can_access_graph: true,
    folder_id: null,
    ...overrides,
  };
}

function setupHandlers({
  agents = [makeAgent()],
  favorites = [] as Array<Record<string, unknown>>,
  folders = [] as Array<Record<string, unknown>>,
  executions = [] as Array<Record<string, unknown>>,
} = {}) {
  server.use(
    http.get(AGENTS_URL, () =>
      HttpResponse.json({
        agents,
        total_count: agents.length,
        page: 1,
        page_size: 20,
        total_pages: 1,
      }),
    ),
    http.get(FAVORITE_AGENTS_URL, () =>
      HttpResponse.json({
        agents: favorites,
        total_count: favorites.length,
        page: 1,
        page_size: 10,
        total_pages: 1,
      }),
    ),
    http.get(FOLDERS_URL, () => HttpResponse.json({ folders })),
    http.get(EXECUTIONS_URL, () => HttpResponse.json(executions)),
  );
}

function waitForAgentsToLoad() {
  return screen.findAllByTestId("library-agent-card-name");
}

describe("LibraryPage", () => {
  test("renders agent cards from API", async () => {
    setupHandlers({ agents: [makeAgent({ name: "Weather Bot" })] });

    render(<LibraryPage />);

    expect(await screen.findByText("Weather Bot")).toBeDefined();
  });

  test("renders multiple agent cards with correct names", async () => {
    setupHandlers({
      agents: [
        makeAgent({ id: "a1", name: "Agent Alpha" }),
        makeAgent({ id: "a2", name: "Agent Beta" }),
        makeAgent({ id: "a3", name: "Agent Gamma" }),
      ],
    });

    render(<LibraryPage />);

    expect(await screen.findByText("Agent Alpha")).toBeDefined();
    expect(screen.getByText("Agent Beta")).toBeDefined();
    expect(screen.getByText("Agent Gamma")).toBeDefined();
  });

  test("renders All and Favorites tabs", async () => {
    setupHandlers();

    render(<LibraryPage />);

    await waitForAgentsToLoad();

    const tabs = screen.getAllByRole("tab");
    const tabNames = tabs.map((t) => t.textContent);
    expect(tabNames.some((n) => n?.match(/all/i))).toBe(true);
    expect(tabNames.some((n) => n?.match(/favorites/i))).toBe(true);
  });

  test("favorites tab is disabled when no favorites exist", async () => {
    setupHandlers();

    render(<LibraryPage />);

    await waitForAgentsToLoad();

    const favoritesTab = screen
      .getAllByRole("tab")
      .find((t) => t.textContent?.match(/favorites/i));
    expect(favoritesTab).toBeDefined();
    expect(favoritesTab!.hasAttribute("data-disabled")).toBe(true);
  });

  test("renders folders alongside agents", async () => {
    setupHandlers({
      folders: [
        {
          id: "f1",
          name: "Work Agents",
          agent_count: 3,
          color: null,
          icon: null,
        },
        {
          id: "f2",
          name: "Personal",
          agent_count: 1,
          color: null,
          icon: null,
        },
      ],
    });

    render(<LibraryPage />);

    expect(await screen.findByText("Work Agents")).toBeDefined();
    expect(screen.getByText("Personal")).toBeDefined();
    expect(screen.getAllByTestId("library-folder")).toHaveLength(2);
  });

  test("shows See runs link on agent card", async () => {
    setupHandlers({
      agents: [makeAgent({ name: "Linked Agent", can_access_graph: true })],
    });

    render(<LibraryPage />);

    await screen.findByText("Linked Agent");

    const runLinks = screen.getAllByText("See runs");
    expect(runLinks.length).toBeGreaterThan(0);
  });

  test("renders search bar and import button", async () => {
    setupHandlers();

    render(<LibraryPage />);

    await waitForAgentsToLoad();

    const searchBars = screen.getAllByTestId("library-textbox");
    expect(searchBars.length).toBeGreaterThan(0);

    const importButtons = screen.getAllByTestId("import-button");
    expect(importButtons.length).toBeGreaterThan(0);
  });

  test("renders Jump Back In when there is an active execution", async () => {
    setupHandlers({
      agents: [
        makeAgent({ id: "lib-1", graph_id: "g-1", name: "Running Agent" }),
      ],
      executions: [
        {
          id: "exec-1",
          graph_id: "g-1",
          graph_version: 1,
          status: "RUNNING",
          started_at: new Date(Date.now() - 60_000).toISOString(),
          ended_at: null,
          stats: null,
        },
      ],
    });

    render(<LibraryPage />);

    expect(await screen.findByText("Jump Back In")).toBeDefined();
  });
});
