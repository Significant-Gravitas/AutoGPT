import {
  getGetV2ListFavoriteLibraryAgentsMockHandler,
  getGetV2ListFavoriteLibraryAgentsResponseMock,
  getGetV2ListLibraryAgentsMockHandler,
  getGetV2ListLibraryAgentsResponseMock,
} from "@/app/api/__generated__/endpoints/library/library.msw";
import { getGetV2ListLibraryFoldersMockHandler } from "@/app/api/__generated__/endpoints/folders/folders.msw";
import { getGetV1ListAllExecutionsMockHandler } from "@/app/api/__generated__/endpoints/graphs/graphs.msw";
import { server } from "@/mocks/mock-server";
import { render, screen } from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { expect, test } from "vitest";
import LibraryPage from "../page";

test("Edit workflow on a library card opens the workflow's active version in the builder", async () => {
  const agent = {
    ...getGetV2ListLibraryAgentsResponseMock().agents[0],
    graph_id: "graph-1",
    // The version the list loaded; a save since then made a newer one.
    graph_version: 3,
    name: "Morning digest",
    can_access_graph: true,
  };
  const page = { total_pages: 1, current_page: 1 };
  server.use(
    getGetV2ListLibraryAgentsMockHandler({
      ...getGetV2ListLibraryAgentsResponseMock(),
      agents: [agent],
      pagination: { ...page, total_items: 1, page_size: 20 },
    }),
    getGetV2ListFavoriteLibraryAgentsMockHandler({
      ...getGetV2ListFavoriteLibraryAgentsResponseMock(),
      agents: [],
      pagination: { ...page, total_items: 0, page_size: 10 },
    }),
    getGetV2ListLibraryFoldersMockHandler({
      folders: [],
      pagination: { ...page, total_items: 0, page_size: 20 },
    }),
    getGetV1ListAllExecutionsMockHandler([]),
  );

  render(<LibraryPage />);

  await screen.findByText("Morning digest");
  await userEvent.click(screen.getByRole("button", { name: "More actions" }));

  const link = await screen.findByTestId(
    "library-agent-card-open-in-builder-link",
  );
  expect(link.getAttribute("href")).toBe("/build?flowID=graph-1");
});
