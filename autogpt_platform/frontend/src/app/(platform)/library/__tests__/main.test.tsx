import { describe, expect, test } from "vitest";
import { render, screen } from "@/tests/integrations/test-utils";
import { server } from "@/mocks/mock-server";
import {
  getGetV2ListLibraryAgentsMockHandler,
  getGetV2ListLibraryAgentsResponseMock,
  getGetV2ListFavoriteLibraryAgentsMockHandler,
  getGetV2ListFavoriteLibraryAgentsResponseMock,
} from "@/app/api/__generated__/endpoints/library/library.msw";
import {
  getGetV2ListLibraryFoldersMockHandler,
  getGetV2ListLibraryFoldersResponseMock,
} from "@/app/api/__generated__/endpoints/folders/folders.msw";
import { getGetV1ListAllExecutionsMockHandler } from "@/app/api/__generated__/endpoints/graphs/graphs.msw";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import LibraryPage from "../page";

function makeAgent(overrides: Partial<LibraryAgent> = {}): LibraryAgent {
  const base = getGetV2ListLibraryAgentsResponseMock().agents[0];
  return { ...base, ...overrides };
}

function setupHandlers({
  agents,
  favorites,
  folders,
  executions,
}: {
  agents?: LibraryAgent[];
  favorites?: LibraryAgent[];
  folders?: Parameters<typeof getGetV2ListLibraryFoldersResponseMock>[0];
  executions?: Parameters<typeof getGetV1ListAllExecutionsMockHandler>[0];
} = {}) {
  const agentList = agents ?? [makeAgent()];
  const favList = favorites ?? [];

  server.use(
    getGetV2ListLibraryAgentsMockHandler({
      ...getGetV2ListLibraryAgentsResponseMock(),
      agents: agentList,
      pagination: {
        total_items: agentList.length,
        total_pages: 1,
        current_page: 1,
        page_size: 20,
      },
    }),
    getGetV2ListFavoriteLibraryAgentsMockHandler({
      ...getGetV2ListFavoriteLibraryAgentsResponseMock(),
      agents: favList,
      pagination: {
        total_items: favList.length,
        total_pages: 1,
        current_page: 1,
        page_size: 10,
      },
    }),
    getGetV2ListLibraryFoldersMockHandler(
      folders
        ? getGetV2ListLibraryFoldersResponseMock(folders)
        : {
            folders: [],
            pagination: {
              total_items: 0,
              total_pages: 1,
              current_page: 1,
              page_size: 20,
            },
          },
    ),
    getGetV1ListAllExecutionsMockHandler(executions ?? []),
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
      folders: {
        folders: [
          {
            id: "f1",
            user_id: "test-user",
            name: "Work Agents",
            agent_count: 3,
            subfolder_count: 0,
            color: null,
            icon: null,
            parent_id: null,
            created_at: new Date(),
            updated_at: new Date(),
          },
          {
            id: "f2",
            user_id: "test-user",
            name: "Personal",
            agent_count: 1,
            subfolder_count: 0,
            color: null,
            icon: null,
            parent_id: null,
            created_at: new Date(),
            updated_at: new Date(),
          },
        ],
      },
    });

    render(<LibraryPage />);

    await waitForAgentsToLoad();

    expect(await screen.findByText("Work Agents")).toBeDefined();
    expect(screen.getByText("Personal")).toBeDefined();
    expect(screen.getAllByTestId("library-folder")).toHaveLength(2);
  });

  test("shows See tasks link on agent card", async () => {
    setupHandlers({
      agents: [makeAgent({ name: "Linked Agent", can_access_graph: true })],
    });

    render(<LibraryPage />);

    await screen.findByText("Linked Agent");

    const runLinks = screen.getAllByText("See tasks");
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

  test("renders running agent card when execution is active", async () => {
    const agent = makeAgent({
      id: "lib-1",
      graph_id: "g-1",
      name: "Running Agent",
    });
    setupHandlers({
      agents: [agent],
      executions: [
        {
          id: "exec-1",
          user_id: "test-user",
          graph_id: "g-1",
          graph_version: 1,
          inputs: {},
          credential_inputs: {},
          nodes_input_masks: {},
          preset_id: null,
          status: "RUNNING",
          started_at: new Date(Date.now() - 60_000),
          ended_at: null,
          stats: null,
        },
      ],
    });

    render(<LibraryPage />);

    expect(await screen.findByText("Running Agent")).toBeDefined();
  });
});
