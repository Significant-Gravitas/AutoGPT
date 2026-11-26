import { QueryClient } from "@tanstack/react-query";
import userEvent from "@testing-library/user-event";
import { http, HttpResponse } from "msw";
import { expect, test } from "vitest";
import {
  getGetV2GetFolderMockHandler,
  getGetV2ListLibraryFoldersMockHandler,
  getPatchV2UpdateFolderMockHandler,
} from "@/app/api/__generated__/endpoints/folders/folders.msw";
import { getGetV1ListAllExecutionsMockHandler } from "@/app/api/__generated__/endpoints/graphs/graphs.msw";
import {
  getGetV2ListFavoriteLibraryAgentsMockHandler,
  getGetV2ListLibraryAgentsMockHandler,
} from "@/app/api/__generated__/endpoints/library/library.msw";
import type { FolderUpdateRequest } from "@/app/api/__generated__/models/folderUpdateRequest";
import type { LibraryFolder } from "@/app/api/__generated__/models/libraryFolder";
import { server } from "@/mocks/mock-server";
import {
  render,
  screen,
  waitFor,
  within,
} from "@/tests/integrations/test-utils";
import LibraryPage from "../page";

const PAGE = { total_pages: 1, current_page: 1, page_size: 20 };

const REPORTS: LibraryFolder = {
  id: "f-reports",
  user_id: "user-1",
  name: "Reports",
  agent_count: 0,
  subfolder_count: 0,
  color: "#3B82F6",
  icon: "📁",
  parent_id: null,
  created_at: new Date(),
  updated_at: new Date(),
};

// One folder, as the backend keeps it: a save changes what every later read
// returns, unless the backend refuses the new name.
function serveFolder({ refuseRename = false } = {}) {
  let stored = REPORTS;
  server.use(
    getGetV2ListLibraryAgentsMockHandler({
      agents: [],
      pagination: { ...PAGE, total_items: 0 },
    }),
    getGetV2ListFavoriteLibraryAgentsMockHandler({
      agents: [],
      pagination: { ...PAGE, total_items: 0 },
    }),
    getGetV1ListAllExecutionsMockHandler([]),
    getGetV2ListLibraryFoldersMockHandler(({ request }) => {
      const listed = new URL(request.url).searchParams.get("parent_id")
        ? []
        : [stored];
      return {
        folders: listed,
        pagination: { ...PAGE, total_items: listed.length },
      };
    }),
    getGetV2GetFolderMockHandler(() => stored),
    refuseRename
      ? http.patch("/api/proxy/api/library/folders/:folderId", () =>
          HttpResponse.json(
            { detail: "A folder with this name already exists" },
            { status: 409 },
          ),
        )
      : getPatchV2UpdateFolderMockHandler(async ({ request }) => {
          const edit = (await request.json()) as FolderUpdateRequest;
          stored = {
            ...stored,
            name: edit.name ?? stored.name,
            color: edit.color ?? stored.color,
            icon: edit.icon ?? stored.icon,
          };
          return stored;
        }),
  );
}

// The app's own stale time (lib/react-query/queryClient.ts): a folder read in
// the last minute isn't read again when it is opened.
function renderLibrary() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false, staleTime: 60_000 } },
  });
  render(<LibraryPage />, { queryClient });
}

// "My Library / <folder>", above an open folder.
function header() {
  return screen.getByRole("button", { name: "My Library" }).parentElement!;
}

async function openFolderThenGoBack(user: ReturnType<typeof userEvent.setup>) {
  await user.click(await screen.findByText("Reports"));
  await waitFor(() => expect(header().textContent).toContain("Reports"));
  await user.click(screen.getByRole("button", { name: "My Library" }));
}

async function renameFolder(
  user: ReturnType<typeof userEvent.setup>,
  name: string,
) {
  await user.click(await screen.findByRole("button", { name: "Edit folder" }));
  const dialog = await screen.findByRole("dialog", { name: "Edit Folder" });
  const field = within(dialog).getByRole("textbox", { name: "Folder name" });
  await user.clear(field);
  await user.type(field, name);
  await user.click(
    within(dialog).getByRole("button", { name: "Save Changes" }),
  );
  return dialog;
}

test("a folder renamed after it was opened opens under its new name", async () => {
  serveFolder();
  const user = userEvent.setup();
  renderLibrary();

  await openFolderThenGoBack(user);
  await renameFolder(user, "Quarterly reports");
  await waitFor(() =>
    expect(screen.queryByRole("dialog", { name: "Edit Folder" })).toBeNull(),
  );

  await user.click(await screen.findByText("Quarterly reports"));
  await waitFor(() =>
    expect(header().textContent).toContain("Quarterly reports"),
  );
  expect(header().textContent).not.toContain("/Reports");
});

test("a refused rename leaves the folder opening under its old name", async () => {
  serveFolder({ refuseRename: true });
  const user = userEvent.setup();
  renderLibrary();

  await openFolderThenGoBack(user);
  const dialog = await renameFolder(user, "Taken name");
  expect(
    await within(dialog).findByText("A folder with this name already exists"),
  ).toBeDefined();
  await user.keyboard("{Escape}");
  await waitFor(() =>
    expect(screen.queryByRole("dialog", { name: "Edit Folder" })).toBeNull(),
  );

  await user.click(await screen.findByText("Reports"));
  await waitFor(() => expect(header().textContent).toContain("Reports"));
  expect(header().textContent).not.toContain("Taken name");
});
