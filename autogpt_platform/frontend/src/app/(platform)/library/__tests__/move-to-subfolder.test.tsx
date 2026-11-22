import { QueryClient } from "@tanstack/react-query";
import userEvent from "@testing-library/user-event";
import { expect, test } from "vitest";
import {
  getGetV2GetFolderTreeMockHandler,
  getGetV2ListLibraryFoldersMockHandler,
  getPostV2BulkMoveAgentsMockHandler,
} from "@/app/api/__generated__/endpoints/folders/folders.msw";
import { getGetV1ListAllExecutionsMockHandler } from "@/app/api/__generated__/endpoints/graphs/graphs.msw";
import {
  getGetV2ListFavoriteLibraryAgentsMockHandler,
  getGetV2ListLibraryAgentsMockHandler,
  getGetV2ListLibraryAgentsResponseMock,
} from "@/app/api/__generated__/endpoints/library/library.msw";
import type { BulkMoveAgentsRequest } from "@/app/api/__generated__/models/bulkMoveAgentsRequest";
import type { LibraryFolder } from "@/app/api/__generated__/models/libraryFolder";
import type { LibraryFolderTree } from "@/app/api/__generated__/models/libraryFolderTree";
import { server } from "@/mocks/mock-server";
import {
  fireEvent,
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import LibraryPage from "../page";

const PAGE = { total_pages: 1, current_page: 1, page_size: 20 };

function folder(id: string, name: string, parentID: string | null = null) {
  return {
    id,
    user_id: "user-1",
    name,
    agent_count: 0,
    subfolder_count: 0,
    color: "#3B82F6",
    icon: "📁",
    parent_id: parentID,
    created_at: new Date(),
    updated_at: new Date(),
  } satisfies LibraryFolder;
}

const WORKFLOW = {
  ...getGetV2ListLibraryAgentsResponseMock().agents[0],
  id: "lib-loose",
  name: "Loose workflow",
  folder_id: null,
};

// The backend's folder routes over a mutable set of folders: the list
// answers one level (the top one without a parent_id), the tree all of them,
// once `treeReady` settles.
function serveFolders(
  folders: LibraryFolder[],
  treeReady: Promise<void> = Promise.resolve(),
) {
  const moves: BulkMoveAgentsRequest[] = [];
  const treeUnder = (parentID: string | null): LibraryFolderTree[] =>
    folders
      .filter((f) => (f.parent_id ?? null) === parentID)
      .map((f) => ({ ...f, children: treeUnder(f.id) }));
  server.use(
    getGetV2ListLibraryAgentsMockHandler({
      agents: [WORKFLOW],
      pagination: { ...PAGE, total_items: 1 },
    }),
    getGetV2ListFavoriteLibraryAgentsMockHandler({
      agents: [],
      pagination: { ...PAGE, total_items: 0 },
    }),
    getGetV1ListAllExecutionsMockHandler([]),
    getGetV2GetFolderTreeMockHandler(async () => {
      await treeReady;
      return { tree: treeUnder(null) };
    }),
    getGetV2ListLibraryFoldersMockHandler(({ request }) => {
      const listed = treeUnder(
        new URL(request.url).searchParams.get("parent_id"),
      );
      return {
        folders: listed,
        pagination: { ...PAGE, total_items: listed.length },
      };
    }),
    getPostV2BulkMoveAgentsMockHandler(async ({ request }) => {
      const move = (await request.json()) as BulkMoveAgentsRequest;
      moves.push(move);
      return [{ ...WORKFLOW, folder_id: move.folder_id }];
    }),
  );
  return moves;
}

async function openMoveDialog() {
  const card = (await screen.findByText("Loose workflow")).closest(
    "[data-testid='library-agent-card']",
  ) as HTMLElement;
  fireEvent.pointerDown(
    within(card).getByRole("button", { name: "More actions" }),
    { button: 0 },
  );
  fireEvent.click(
    await screen.findByRole("menuitem", { name: "Move to folder" }),
  );
  return screen.findByRole("dialog", { name: "Move to folder" });
}

test("Move to folder lists a subfolder under its parent and moves a workflow into it", async () => {
  const moves = serveFolders([
    folder("f-clients", "Clients"),
    folder("f-acme", "Acme", "f-clients"),
  ]);
  const user = userEvent.setup();
  render(<LibraryPage />);

  const dialog = await openMoveDialog();
  await within(dialog).findByText("Clients");
  const subfolder = await within(dialog).findByRole("button", {
    name: /^📁\s*Acme\b/,
  });
  const listed = within(dialog)
    .getAllByRole("button", { name: /^📁/ })
    .map((button) => button.textContent ?? "");
  const parentAt = listed.findIndex((name) => name.includes("Clients"));
  expect(parentAt).toBeGreaterThanOrEqual(0);
  expect(listed.findIndex((name) => name.includes("Acme"))).toBeGreaterThan(
    parentAt,
  );
  expect(subfolder.style.getPropertyValue("--folder-depth")).toBe("1");
  await user.click(subfolder);

  await waitFor(() =>
    expect(moves).toEqual([{ agent_ids: ["lib-loose"], folder_id: "f-acme" }]),
  );
});

test("Move to folder opens on the folders as they are, with the app's stale time", async () => {
  const folders = [folder("f-clients", "Clients")];
  serveFolders(folders);
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false, staleTime: 60_000 } },
  });
  const user = userEvent.setup();
  render(<LibraryPage />, { queryClient });

  let dialog = await openMoveDialog();
  await within(dialog).findByText("Clients");
  await user.keyboard("{Escape}");
  await waitFor(() =>
    expect(screen.queryByRole("dialog", { name: "Move to folder" })).toBeNull(),
  );

  // Made since the dialog was last open, the way Otto makes one: nothing on
  // this page knew to refresh its folders.
  folders.push(folder("f-acme", "Acme", "f-clients"));
  dialog = await openMoveDialog();
  expect(
    await within(dialog).findByRole("button", { name: /^📁\s*Acme\b/ }),
  ).toBeDefined();
});

test("Move to folder says it is loading, not that there are no folders, until they arrive", async () => {
  let releaseTree: () => void = () => {};
  serveFolders(
    [folder("f-clients", "Clients")],
    new Promise<void>((resolve) => {
      releaseTree = () => resolve();
    }),
  );
  render(<LibraryPage />);

  try {
    const dialog = await openMoveDialog();
    expect(await within(dialog).findByText("Loading folders…")).toBeDefined();
    expect(within(dialog).queryByText("No folders found")).toBeNull();
    releaseTree();
    expect(await within(dialog).findByText("Clients")).toBeDefined();
  } finally {
    releaseTree();
  }
});
