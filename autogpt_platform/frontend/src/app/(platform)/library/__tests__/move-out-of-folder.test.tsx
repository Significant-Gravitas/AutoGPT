import userEvent from "@testing-library/user-event";
import { NuqsTestingAdapter } from "nuqs/adapters/testing";
import { expect, test } from "vitest";
import {
  getGetV2GetFolderMockHandler,
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

function folder(id: string, name: string): LibraryFolder {
  return {
    id,
    user_id: "user-1",
    name,
    agent_count: 1,
    subfolder_count: 0,
    color: "#3B82F6",
    icon: "📁",
    parent_id: null,
    created_at: new Date(),
    updated_at: new Date(),
  };
}

const INVOICES = folder("f-invoices", "Invoices");

function workflow(folderID: string | null) {
  return {
    ...getGetV2ListLibraryAgentsResponseMock().agents[0],
    id: "lib-billing",
    name: "Billing run",
    folder_id: folderID,
  };
}

// Top-level folders, by both the folder list and the folder tree, and the
// one workflow, which the list shows in whichever folder is open.
function serveLibrary(folders: LibraryFolder[], folderID: string | null) {
  const moves: BulkMoveAgentsRequest[] = [];
  server.use(
    getGetV2ListLibraryAgentsMockHandler(({ request }) => {
      const open = new URL(request.url).searchParams.get("folder_id");
      const agents = open === folderID ? [workflow(folderID)] : [];
      return { agents, pagination: { ...PAGE, total_items: agents.length } };
    }),
    getGetV2ListFavoriteLibraryAgentsMockHandler({
      agents: [],
      pagination: { ...PAGE, total_items: 0 },
    }),
    getGetV1ListAllExecutionsMockHandler([]),
    getGetV2GetFolderTreeMockHandler({
      tree: folders.map((f) => ({ ...f, children: [] })),
    }),
    getGetV2GetFolderMockHandler(({ params }) => {
      const found = folders.find((f) => f.id === params.folderId);
      if (!found) throw new Error(`no folder ${String(params.folderId)}`);
      return found;
    }),
    getGetV2ListLibraryFoldersMockHandler(({ request }) => {
      const listed = new URL(request.url).searchParams.get("parent_id")
        ? []
        : folders;
      return {
        folders: listed,
        pagination: { ...PAGE, total_items: listed.length },
      };
    }),
    getPostV2BulkMoveAgentsMockHandler(async ({ request }) => {
      const move = (await request.json()) as BulkMoveAgentsRequest;
      moves.push(move);
      return [workflow(move.folder_id ?? null)];
    }),
  );
  return moves;
}

function renderLibrary(searchParams = "") {
  render(
    <NuqsTestingAdapter searchParams={searchParams}>
      <LibraryPage />
    </NuqsTestingAdapter>,
  );
}

async function openMoveDialog() {
  const card = (await screen.findByText("Billing run")).closest(
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

function myLibraryOption(dialog: HTMLElement) {
  return within(dialog).queryByRole("button", {
    name: /^📂\s*My Library \(root\)/,
  });
}

test("a workflow in the only folder can be moved back to My Library", async () => {
  const moves = serveLibrary([INVOICES], INVOICES.id);
  const user = userEvent.setup();
  renderLibrary(`?folder=${INVOICES.id}`);

  const dialog = await openMoveDialog();
  await waitFor(() => expect(myLibraryOption(dialog)).not.toBeNull());
  await user.click(myLibraryOption(dialog)!);

  await waitFor(() =>
    expect(moves).toEqual([{ agent_ids: ["lib-billing"], folder_id: null }]),
  );
});

test("a folder search that matches nothing still offers My Library", async () => {
  serveLibrary([INVOICES, folder("f-receipts", "Receipts")], INVOICES.id);
  const user = userEvent.setup();
  renderLibrary(`?folder=${INVOICES.id}`);

  const dialog = await openMoveDialog();
  await within(dialog).findByText("Receipts");
  await user.type(
    within(dialog).getByRole("textbox", { name: "Search folders" }),
    "no such folder",
  );

  expect(await within(dialog).findByText("No folders found")).toBeDefined();
  expect(myLibraryOption(dialog)).not.toBeNull();
});

test("a workflow already in My Library isn't offered My Library", async () => {
  serveLibrary([INVOICES], null);
  renderLibrary();

  const dialog = await openMoveDialog();
  await within(dialog).findByText("Invoices");
  expect(myLibraryOption(dialog)).toBeNull();
});
