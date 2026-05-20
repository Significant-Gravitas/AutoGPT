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
  folders,
}: {
  agents?: LibraryAgent[];
  folders?: Parameters<typeof getGetV2ListLibraryFoldersResponseMock>[0];
} = {}) {
  const agentList = agents ?? [];

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
      agents: [],
      pagination: {
        total_items: 0,
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
    getGetV1ListAllExecutionsMockHandler([]),
  );
}

describe("LibraryPage empty state", () => {
  test("renders empty-state heading and copy when no agents and no folders", async () => {
    setupHandlers();

    render(<LibraryPage />);

    expect(await screen.findByText("Your library is empty")).toBeDefined();
    expect(
      screen.getByText(/build your own agent from scratch/i),
    ).toBeDefined();
  });

  test("renders Build an agent CTA pointing to /build", async () => {
    setupHandlers();

    render(<LibraryPage />);

    const buildLink = await screen.findByRole("link", {
      name: /build an agent/i,
    });
    expect(buildLink.getAttribute("href")).toBe("/build");
  });

  test("renders Browse marketplace CTA pointing to /marketplace", async () => {
    setupHandlers();

    render(<LibraryPage />);

    const marketplaceLink = await screen.findByRole("link", {
      name: /browse marketplace/i,
    });
    expect(marketplaceLink.getAttribute("href")).toBe("/marketplace");
  });

  test("does not render empty state when at least one agent exists", async () => {
    setupHandlers({ agents: [makeAgent({ name: "Existing Agent" })] });

    render(<LibraryPage />);

    expect(await screen.findByText("Existing Agent")).toBeDefined();
    expect(screen.queryByText("Your library is empty")).toBeNull();
  });

  test("does not render empty state when folders exist but no agents", async () => {
    setupHandlers({
      folders: {
        folders: [
          {
            id: "f1",
            user_id: "test-user",
            name: "My Folder",
            agent_count: 0,
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

    expect(await screen.findByText("My Folder")).toBeDefined();
    expect(screen.queryByText("Your library is empty")).toBeNull();
  });
});
