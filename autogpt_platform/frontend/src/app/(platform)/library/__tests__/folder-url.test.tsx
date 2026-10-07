import userEvent from "@testing-library/user-event";
import { NuqsTestingAdapter, type UrlUpdateEvent } from "nuqs/adapters/testing";
import { expect, test, vi } from "vitest";
import { getGetV1ListAllExecutionsMockHandler } from "@/app/api/__generated__/endpoints/graphs/graphs.msw";
import {
  getGetV2GetFolderMockHandler,
  getGetV2GetFolderResponseMock,
  getGetV2ListLibraryFoldersMockHandler,
} from "@/app/api/__generated__/endpoints/folders/folders.msw";
import {
  getGetV2ListFavoriteLibraryAgentsMockHandler,
  getGetV2ListLibraryAgentsMockHandler,
  getGetV2ListLibraryAgentsResponseMock,
} from "@/app/api/__generated__/endpoints/library/library.msw";
import type { LibraryFolder } from "@/app/api/__generated__/models/libraryFolder";
import { server } from "@/mocks/mock-server";
import { render, screen, waitFor } from "@/tests/integrations/test-utils";
import LibraryPage from "../page";

const FOLDER: LibraryFolder = {
  id: "f-q3",
  user_id: "test-user",
  name: "Q3 reports",
  agent_count: 0,
  subfolder_count: 0,
  color: null,
  icon: null,
  parent_id: null,
  created_at: new Date(),
  updated_at: new Date(),
};
const PAGE = { total_items: 0, total_pages: 1, current_page: 1, page_size: 20 };

function serve() {
  const agentQueries: URLSearchParams[] = [];
  server.use(
    getGetV2ListLibraryAgentsMockHandler(({ request }) => {
      agentQueries.push(new URL(request.url).searchParams);
      return {
        ...getGetV2ListLibraryAgentsResponseMock(),
        agents: [],
        pagination: PAGE,
      };
    }),
    getGetV2ListFavoriteLibraryAgentsMockHandler({
      agents: [],
      pagination: PAGE,
    }),
    getGetV2ListLibraryFoldersMockHandler({
      folders: [FOLDER],
      pagination: { ...PAGE, total_items: 1 },
    }),
    getGetV2GetFolderMockHandler(getGetV2GetFolderResponseMock(FOLDER)),
    getGetV1ListAllExecutionsMockHandler([]),
  );
  return agentQueries;
}

test("a library link with ?folder= opens that folder", async () => {
  const agentQueries = serve();

  render(
    <NuqsTestingAdapter searchParams="?folder=f-q3">
      <LibraryPage />
    </NuqsTestingAdapter>,
  );

  await waitFor(() =>
    expect(agentQueries.some((q) => q.get("folder_id") === "f-q3")).toBe(true),
  );
  expect(agentQueries.some((q) => q.get("include_root_only"))).toBe(false);
});

test("opening a folder puts it in the URL", async () => {
  serve();
  const onUrlUpdate = vi.fn<(event: UrlUpdateEvent) => void>();

  render(
    <NuqsTestingAdapter onUrlUpdate={onUrlUpdate}>
      <LibraryPage />
    </NuqsTestingAdapter>,
  );
  await userEvent.click(await screen.findByTestId("library-folder"));

  await waitFor(() =>
    expect(onUrlUpdate.mock.calls.at(-1)?.[0].queryString).toBe("?folder=f-q3"),
  );
});
