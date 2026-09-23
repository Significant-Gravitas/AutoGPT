import { beforeEach, describe, expect, test, vi } from "vitest";

import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import userEvent from "@testing-library/user-event";
import { server } from "@/mocks/mock-server";
import { currentFolderParam, resetNavigation } from "./navigation-mock";
import { http, HttpResponse } from "msw";
import { getListExpertIdentitiesMockHandler } from "@/app/api/__generated__/endpoints/experts/experts.msw";
import {
  getGetWorkspaceStorageUsageMockHandler,
  getListWorkspaceFilesMockHandler,
  getListWorkspaceFoldersMockHandler,
} from "@/app/api/__generated__/endpoints/workspace/workspace.msw";
import type { WorkspaceFileItem } from "@/app/api/__generated__/models/workspaceFileItem";
import type { WorkspaceFolder } from "@/app/api/__generated__/models/workspaceFolder";
import { FILE_DRAG_MIME } from "../components/WorkspaceFolders/drag";

vi.mock("@/services/feature-flags/use-get-flag", () => ({
  Flag: { ARTIFACTS_PAGE: "artifacts-page" },
  useGetFlag: () => true,
  useFlagStatus: () => ({ enabled: true, ready: true }),
}));

// The generated default answers with random experts, which would render
// random filter tabs on the page under test.
beforeEach(() => {
  resetNavigation();
  server.use(getListExpertIdentitiesMockHandler([]));
});

vi.mock("next/navigation", async () => {
  const { navigationMock } = await import("./navigation-mock");
  return navigationMock();
});

vi.mock("framer-motion", async (importActual) => {
  const actual = await importActual<typeof import("framer-motion")>();
  return { ...actual, useReducedMotion: () => true };
});

import ArtifactsPage from "../page";

const PROXY = "/api/proxy/api/workspace";

function makeFile(
  overrides: Partial<WorkspaceFileItem> = {},
): WorkspaceFileItem {
  return {
    id: "file-base",
    name: "base.txt",
    path: "/base.txt",
    mime_type: "text/plain",
    size_bytes: 1024,
    folder_id: null,
    metadata: {},
    origin: "generated",
    created_at: "2026-05-01T00:00:00Z",
    ...overrides,
  };
}

function makeFolder(overrides: Partial<WorkspaceFolder> = {}): WorkspaceFolder {
  return {
    id: "fld-1",
    workspace_id: "ws-1",
    name: "Reports",
    file_count: 2,
    created_at: "2026-05-01T00:00:00Z" as unknown as Date,
    updated_at: "2026-05-01T00:00:00Z" as unknown as Date,
    ...overrides,
  };
}

function useStorageHandler() {
  server.use(
    getGetWorkspaceStorageUsageMockHandler({
      used_bytes: 0,
      limit_bytes: 1_000_000_000,
      used_percent: 0,
      file_count: 0,
    }),
  );
}

// "New folder" lives in the header's New menu. Radix DropdownMenu opens on
// pointerdown, not click, under happy-dom.
async function openCreateFolderDialog() {
  fireEvent.pointerDown(screen.getByTestId("artifacts-new-menu"), {
    button: 0,
  });
  fireEvent.click(await screen.findByTestId("create-folder-button"));
}

// Folder actions live behind a "…" menu, like the file rows'. Radix
// DropdownMenu opens on pointerdown, not click, under happy-dom.
async function openFolderMenu(folderName = "Reports") {
  fireEvent.pointerDown(
    await screen.findByLabelText(`Actions for ${folderName}`),
    { button: 0 },
  );
}

// The move dialog selects first and moves on confirm, so a mis-click on a
// chevron can never move anything.
async function chooseMoveTarget(name: string) {
  const rows = await screen.findAllByTestId("move-to-folder-option");
  const row = rows.find((r) => r.textContent?.includes(name));
  if (!row) throw new Error(`no move target named ${name}`);
  fireEvent.click(row);
  fireEvent.click(screen.getByTestId("confirm-move-to-folder"));
}

describe("ArtifactsPage - folders", () => {
  test("renders folders as rows at the root with file counts", async () => {
    useStorageHandler();
    server.use(
      getListWorkspaceFilesMockHandler({
        files: [],
        offset: 0,
        has_more: false,
      }),
      getListWorkspaceFoldersMockHandler({
        folders: [makeFolder({ name: "Reports", file_count: 2 })],
      }),
    );

    render(<ArtifactsPage />);

    expect(await screen.findByTestId("workspace-folder")).toBeDefined();
    expect(screen.getByText("Reports")).toBeDefined();
    expect(screen.getByText("2 files")).toBeDefined();
  });

  test("grid view shows folders as cards", async () => {
    useStorageHandler();
    server.use(
      getListWorkspaceFilesMockHandler({
        files: [],
        offset: 0,
        has_more: false,
      }),
      getListWorkspaceFoldersMockHandler({
        folders: [makeFolder({ name: "Reports", file_count: 2 })],
      }),
    );

    render(<ArtifactsPage />);

    fireEvent.click(await screen.findByTestId("artifacts-view-grid"));

    expect(await screen.findByTestId("workspace-folders")).toBeDefined();
    expect(screen.getByText("Reports")).toBeDefined();
  });

  test("selecting a folder scopes the list and shows a breadcrumb", async () => {
    useStorageHandler();
    server.use(
      getListWorkspaceFoldersMockHandler({
        folders: [makeFolder({ id: "fld-1", name: "Reports" })],
      }),
      // Branch on folder_id: root shows root.txt, folder shows inside.txt.
      http.get(`${PROXY}/files`, ({ request }) => {
        const url = new URL(request.url);
        const inFolder = url.searchParams.get("folder_id") === "fld-1";
        return HttpResponse.json({
          files: [
            inFolder
              ? makeFile({ id: "f-in", name: "inside.txt", folder_id: "fld-1" })
              : makeFile({ id: "f-root", name: "root.txt" }),
          ],
          offset: 0,
          has_more: false,
        });
      }),
    );

    render(<ArtifactsPage />);

    expect(await screen.findByText("root.txt")).toBeDefined();

    fireEvent.click(await screen.findByTestId("workspace-folder"));

    expect(await screen.findByTestId("folder-breadcrumb")).toBeDefined();
    expect(await screen.findByText("inside.txt")).toBeDefined();
    // Folder rows only appear at the root.
    expect(screen.queryByTestId("workspace-folder")).toBeNull();

    // Back to root via breadcrumb.
    fireEvent.click(screen.getByTestId("folder-breadcrumb-root"));
    expect(await screen.findByText("root.txt")).toBeDefined();
  });

  test("creating a folder posts to the create endpoint", async () => {
    useStorageHandler();
    let createdName: string | null = null;
    server.use(
      getListWorkspaceFilesMockHandler({
        files: [],
        offset: 0,
        has_more: false,
      }),
      getListWorkspaceFoldersMockHandler({ folders: [] }),
      http.post(`${PROXY}/folders`, async ({ request }) => {
        const body = (await request.json()) as { name: string };
        createdName = body.name;
        return HttpResponse.json(makeFolder({ name: body.name }), {
          status: 201,
        });
      }),
    );

    render(<ArtifactsPage />);

    await screen.findByTestId("artifacts-empty");
    await openCreateFolderDialog();
    fireEvent.change(await screen.findByLabelText(/folder name/i), {
      target: { value: "Invoices" },
    });
    fireEvent.click(screen.getByTestId("folder-form-submit"));

    await waitFor(() => expect(createdName).toBe("Invoices"));
  });

  test("move-to-folder menu posts a bulk move", async () => {
    useStorageHandler();
    let movedTo: string | null | undefined;
    server.use(
      getListWorkspaceFilesMockHandler({
        files: [makeFile({ id: "f1", name: "movable.txt" })],
        offset: 0,
        has_more: false,
      }),
      getListWorkspaceFoldersMockHandler({
        folders: [makeFolder({ id: "fld-1", name: "Reports" })],
      }),
      http.post(`${PROXY}/folders/files/bulk-move`, async ({ request }) => {
        const body = (await request.json()) as { folder_id: string | null };
        movedTo = body.folder_id;
        return HttpResponse.json([]);
      }),
    );

    render(<ArtifactsPage />);

    expect(await screen.findByText("movable.txt")).toBeDefined();
    fireEvent.pointerDown(screen.getByTestId("artifacts-card-menu"), {
      button: 0,
    });
    fireEvent.click(await screen.findByTestId("artifacts-move-to-folder"));
    await chooseMoveTarget("Reports");

    await waitFor(() => expect(movedTo).toBe("fld-1"));
  });

  test("renaming a folder patches the folder", async () => {
    useStorageHandler();
    let patchedName: string | null = null;
    server.use(
      getListWorkspaceFilesMockHandler({
        files: [],
        offset: 0,
        has_more: false,
      }),
      getListWorkspaceFoldersMockHandler({
        folders: [makeFolder({ id: "fld-1", name: "Reports" })],
      }),
      http.patch(`${PROXY}/folders/fld-1`, async ({ request }) => {
        const body = (await request.json()) as { name?: string };
        patchedName = body.name ?? null;
        return HttpResponse.json(
          makeFolder({ id: "fld-1", name: body.name ?? "Reports" }),
        );
      }),
    );

    render(<ArtifactsPage />);

    await openFolderMenu();
    fireEvent.click(await screen.findByTestId("folder-rename-menu"));
    const input = await screen.findByLabelText(/folder name/i);
    fireEvent.change(input, { target: { value: "Invoices" } });
    // Submit via Enter to exercise the input's keydown handler.
    fireEvent.keyDown(input, { key: "Enter" });

    await waitFor(() => expect(patchedName).toBe("Invoices"));
  });

  test("deleting a folder calls the delete endpoint", async () => {
    useStorageHandler();
    let deleted = false;
    server.use(
      getListWorkspaceFilesMockHandler({
        files: [],
        offset: 0,
        has_more: false,
      }),
      getListWorkspaceFoldersMockHandler({
        folders: [makeFolder({ id: "fld-1", name: "Reports" })],
      }),
      http.delete(`${PROXY}/folders/fld-1`, () => {
        deleted = true;
        return new HttpResponse(null, { status: 204 });
      }),
    );

    render(<ArtifactsPage />);

    await openFolderMenu();
    fireEvent.click(await screen.findByTestId("folder-delete-menu"));
    fireEvent.click(await screen.findByTestId("confirm-delete-folder"));

    await waitFor(() => expect(deleted).toBe(true));
  });

  test("move-to-folder offers root when the file is already in a folder", async () => {
    useStorageHandler();
    let movedTo: string | null | undefined;
    server.use(
      getListWorkspaceFilesMockHandler({
        files: [
          makeFile({ id: "f1", name: "movable.txt", folder_id: "fld-1" }),
        ],
        offset: 0,
        has_more: false,
      }),
      getListWorkspaceFoldersMockHandler({
        folders: [makeFolder({ id: "fld-1", name: "Reports" })],
      }),
      http.post(`${PROXY}/folders/files/bulk-move`, async ({ request }) => {
        const body = (await request.json()) as { folder_id: string | null };
        movedTo = body.folder_id;
        return HttpResponse.json([]);
      }),
    );

    render(<ArtifactsPage />);

    expect(await screen.findByText("movable.txt")).toBeDefined();
    fireEvent.pointerDown(screen.getByTestId("artifacts-card-menu"), {
      button: 0,
    });
    fireEvent.click(await screen.findByTestId("artifacts-move-to-folder"));
    await chooseMoveTarget("Files (root)");

    await waitFor(() => expect(movedTo).toBeNull());
  });

  test("shows an error card when folders fail to load", async () => {
    useStorageHandler();
    server.use(
      getListWorkspaceFilesMockHandler({
        files: [],
        offset: 0,
        has_more: false,
      }),
      http.get(`${PROXY}/folders`, () =>
        HttpResponse.json({ detail: "boom" }, { status: 500 }),
      ),
    );

    render(<ArtifactsPage />);

    expect(await screen.findByText(/something went wrong/i)).toBeDefined();
  });

  test("dragging a file onto a folder row moves it", async () => {
    useStorageHandler();
    let movedTo: string | null | undefined;
    server.use(
      getListWorkspaceFilesMockHandler({
        files: [makeFile({ id: "f1", name: "drag.txt" })],
        offset: 0,
        has_more: false,
      }),
      getListWorkspaceFoldersMockHandler({
        folders: [makeFolder({ id: "fld-1", name: "Reports" })],
      }),
      http.post(`${PROXY}/folders/files/bulk-move`, async ({ request }) => {
        const body = (await request.json()) as {
          file_ids: string[];
          folder_id: string | null;
        };
        movedTo = body.folder_id;
        return HttpResponse.json([]);
      }),
    );

    render(<ArtifactsPage />);

    expect(await screen.findByText("drag.txt")).toBeDefined();

    const store: Record<string, string> = {};
    const dataTransfer = {
      setData: (key: string, value: string) => {
        store[key] = value;
      },
      getData: (key: string) => store[key] ?? "",
      setDragImage: () => {},
      get types() {
        return Object.keys(store);
      },
    };

    const row = screen.getByTestId("artifacts-list-item");
    const folder = await screen.findByTestId("workspace-folder");
    fireEvent.dragStart(row, { dataTransfer });
    fireEvent.dragOver(folder, { dataTransfer });
    fireEvent.drop(folder, { dataTransfer });
    fireEvent.dragEnd(row, { dataTransfer });

    await waitFor(() => expect(movedTo).toBe("fld-1"));
    expect(store[FILE_DRAG_MIME]).toBe("f1");
  });

  test("opening a folder with the keyboard scopes the list", async () => {
    useStorageHandler();
    server.use(
      getListWorkspaceFoldersMockHandler({
        folders: [makeFolder({ id: "fld-1", name: "Reports" })],
      }),
      http.get(`${PROXY}/files`, ({ request }) => {
        const inFolder =
          new URL(request.url).searchParams.get("folder_id") === "fld-1";
        return HttpResponse.json({
          files: [
            inFolder
              ? makeFile({ id: "f-in", name: "inside.txt", folder_id: "fld-1" })
              : makeFile({ id: "f-root", name: "root.txt" }),
          ],
          offset: 0,
          has_more: false,
        });
      }),
    );

    render(<ArtifactsPage />);

    expect(await screen.findByText("root.txt")).toBeDefined();
    const user = userEvent.setup();
    (await screen.findByTestId("workspace-folder")).focus();
    await user.keyboard("{Enter}");
    expect(await screen.findByText("inside.txt")).toBeDefined();
  });

  test("create failure keeps the dialog open and toasts", async () => {
    useStorageHandler();
    server.use(
      getListWorkspaceFilesMockHandler({
        files: [],
        offset: 0,
        has_more: false,
      }),
      getListWorkspaceFoldersMockHandler({ folders: [] }),
      http.post(`${PROXY}/folders`, () =>
        HttpResponse.json({ detail: "exists" }, { status: 409 }),
      ),
    );

    render(<ArtifactsPage />);

    await screen.findByTestId("artifacts-empty");
    await openCreateFolderDialog();
    fireEvent.change(await screen.findByLabelText(/folder name/i), {
      target: { value: "Dupe" },
    });
    fireEvent.click(screen.getByTestId("folder-form-submit"));

    // Dialog stays open on error (submit button still present).
    await waitFor(() =>
      expect(screen.getByTestId("folder-form-submit")).toBeDefined(),
    );
  });

  test("cancelling the delete dialog closes it without deleting", async () => {
    useStorageHandler();
    let deleteCalled = false;
    server.use(
      getListWorkspaceFilesMockHandler({
        files: [],
        offset: 0,
        has_more: false,
      }),
      getListWorkspaceFoldersMockHandler({
        folders: [makeFolder({ id: "fld-1", name: "Reports" })],
      }),
      http.delete(`${PROXY}/folders/fld-1`, () => {
        deleteCalled = true;
        return new HttpResponse(null, { status: 204 });
      }),
    );

    render(<ArtifactsPage />);

    await openFolderMenu();
    fireEvent.click(await screen.findByTestId("folder-delete-menu"));
    fireEvent.click(await screen.findByRole("button", { name: /cancel/i }));

    await waitFor(() =>
      expect(screen.queryByTestId("confirm-delete-folder")).toBeNull(),
    );
    expect(deleteCalled).toBe(false);
  });
});

describe("ArtifactsPage - nested folders", () => {
  // Files / Reports / 2026, with one file directly in Reports.
  function useNestedHandlers() {
    useStorageHandler();
    server.use(
      getListWorkspaceFoldersMockHandler({
        folders: [
          makeFolder({ id: "fld-1", name: "Reports", file_count: 1 }),
          makeFolder({
            id: "fld-2",
            name: "2026",
            parent_id: "fld-1",
            file_count: 0,
          }),
          makeFolder({ id: "fld-9", name: "Archive", file_count: 0 }),
        ],
      }),
      http.get(`${PROXY}/files`, ({ request }) => {
        const folderId = new URL(request.url).searchParams.get("folder_id");
        return HttpResponse.json({
          files:
            folderId === "fld-1"
              ? [makeFile({ id: "f-in", name: "q3.pdf", folder_id: "fld-1" })]
              : [],
          offset: 0,
          has_more: false,
        });
      }),
    );
  }

  test("inside a folder the rows are its children and its direct files", async () => {
    resetNavigation("folder=fld-1");
    useNestedHandlers();

    render(<ArtifactsPage />);

    expect(await screen.findByText("q3.pdf")).toBeDefined();
    expect(screen.getByText("2026")).toBeDefined();
    // A sibling of the open folder is not one of its children.
    expect(screen.queryByText("Archive")).toBeNull();
  });

  test("grid view inside a folder shows the child folders only", async () => {
    resetNavigation("folder=fld-1");
    useNestedHandlers();

    render(<ArtifactsPage />);
    fireEvent.click(await screen.findByTestId("artifacts-view-grid"));

    expect(await screen.findByText("2026")).toBeDefined();
    expect(screen.queryByText("Archive")).toBeNull();
  });

  test("a folder holding only subfolders reports them instead of 0 files", async () => {
    useStorageHandler();
    server.use(
      getListWorkspaceFilesMockHandler({
        files: [],
        offset: 0,
        has_more: false,
      }),
      getListWorkspaceFoldersMockHandler({
        folders: [
          makeFolder({ id: "fld-1", name: "Reports", file_count: 0 }),
          makeFolder({ id: "fld-2", name: "2026", parent_id: "fld-1" }),
          makeFolder({ id: "fld-9", name: "Archive", file_count: 0 }),
        ],
      }),
    );

    render(<ArtifactsPage />);

    expect(await screen.findByText("1 folder")).toBeDefined();
    expect(screen.getByText("Empty")).toBeDefined();
  });

  // The delete cascades down the whole subtree, so the dialog counts every
  // folder that goes, not the two children the folder rows summarise.
  test("the delete dialog counts the whole subtree, not the direct children", async () => {
    useStorageHandler();
    server.use(
      getListWorkspaceFilesMockHandler({
        files: [],
        offset: 0,
        has_more: false,
      }),
      getListWorkspaceFoldersMockHandler({
        folders: [
          makeFolder({ id: "fld-1", name: "Reports", file_count: 0 }),
          makeFolder({ id: "fld-2", name: "2026", parent_id: "fld-1" }),
          makeFolder({ id: "fld-3", name: "Drafts", parent_id: "fld-1" }),
          makeFolder({ id: "fld-4", name: "Q3", parent_id: "fld-2" }),
        ],
      }),
    );

    render(<ArtifactsPage />);

    await openFolderMenu();
    fireEvent.click(await screen.findByTestId("folder-delete-menu"));

    expect(await screen.findByText(/and its 3 folders\?/)).toBeDefined();
  });

  test("New folder inside a folder posts parent_id and names the destination", async () => {
    resetNavigation("folder=fld-1");
    useNestedHandlers();
    let parentId: string | null | undefined = undefined;
    server.use(
      http.post(`${PROXY}/folders`, async ({ request }) => {
        const body = (await request.json()) as { parent_id?: string | null };
        parentId = body.parent_id;
        return HttpResponse.json(makeFolder({ id: "fld-new", name: "Q4" }), {
          status: 201,
        });
      }),
    );

    render(<ArtifactsPage />);
    await screen.findByText("q3.pdf");
    await openCreateFolderDialog();

    expect(await screen.findByText(/Inside “Reports”/)).toBeDefined();
    fireEvent.change(await screen.findByLabelText(/folder name/i), {
      target: { value: "Q4" },
    });
    fireEvent.click(screen.getByTestId("folder-form-submit"));

    await waitFor(() => expect(parentId).toBe("fld-1"));
  });

  test("the delete dialog names the folders that go with it", async () => {
    useNestedHandlers();

    render(<ArtifactsPage />);
    await openFolderMenu("Reports");
    fireEvent.click(await screen.findByTestId("folder-delete-menu"));

    expect(
      await screen.findByText(/and its 1 folder\? Files inside them/),
    ).toBeDefined();
  });

  test("a folder that vanishes while open lands you in its parent, not at the root", async () => {
    resetNavigation("folder=fld-2");
    useNestedHandlers();
    server.use(
      http.post(`${PROXY}/folders`, () => {
        // Someone else removed "2026" meanwhile; the create's invalidation is
        // what makes this tab notice.
        server.use(
          getListWorkspaceFoldersMockHandler({
            folders: [makeFolder({ id: "fld-1", name: "Reports" })],
          }),
        );
        return HttpResponse.json(makeFolder({ id: "fld-new", name: "Q4" }), {
          status: 201,
        });
      }),
    );

    render(<ArtifactsPage />);
    await screen.findByText("2026");
    await openCreateFolderDialog();
    fireEvent.change(await screen.findByLabelText(/folder name/i), {
      target: { value: "Q4" },
    });
    fireEvent.click(screen.getByTestId("folder-form-submit"));

    await waitFor(() => expect(currentFolderParam()).toBe("fld-1"));
  });

  test("the folder row menu offers rename, move and delete", async () => {
    useNestedHandlers();

    render(<ArtifactsPage />);
    await openFolderMenu("Reports");

    expect(await screen.findByTestId("folder-rename-menu")).toBeDefined();
    expect(screen.getByTestId("folder-move-menu")).toBeDefined();
    expect(screen.getByTestId("folder-delete-menu")).toBeDefined();
  });

  test("moving a folder refuses itself and its descendants, and PATCHes the chosen parent", async () => {
    useNestedHandlers();
    let patched: { id: string; parentId: string | null } | null = null;
    server.use(
      http.patch(`${PROXY}/folders/:folderId`, async ({ request, params }) => {
        const body = (await request.json()) as { parent_id: string | null };
        patched = {
          id: params.folderId as string,
          parentId: body.parent_id,
        };
        return HttpResponse.json(makeFolder({ id: "fld-1" }));
      }),
    );

    render(<ArtifactsPage />);
    await openFolderMenu("Reports");
    fireEvent.click(await screen.findByTestId("folder-move-menu"));

    const rows = await screen.findAllByTestId("move-to-folder-option");
    const self = rows.find((r) => r.textContent?.includes("Reports"));
    expect(self?.getAttribute("aria-disabled")).toBe("true");
    expect(self?.textContent).toContain("Folder being moved");
    // The tree opens on the subject's own location, so its child is already
    // visible — refused too, and it says why.
    const child = rows.find((r) => r.textContent?.includes("2026"));
    expect(child?.getAttribute("aria-disabled")).toBe("true");
    expect(child?.textContent).toContain("Inside the folder being moved");

    const archive = (
      await screen.findAllByTestId("move-to-folder-option")
    ).find((r) => r.textContent?.includes("Archive"));
    fireEvent.click(archive as HTMLElement);
    fireEvent.click(screen.getByTestId("confirm-move-to-folder"));

    await waitFor(() =>
      expect(patched).toEqual({ id: "fld-1", parentId: "fld-9" }),
    );
  });

  test("Move stays disabled until a destination is selected", async () => {
    useNestedHandlers();

    render(<ArtifactsPage />);
    await openFolderMenu("Reports");
    fireEvent.click(await screen.findByTestId("folder-move-menu"));

    const confirm = await screen.findByTestId("confirm-move-to-folder");
    expect(confirm.hasAttribute("disabled")).toBe(true);

    const archive = (
      await screen.findAllByTestId("move-to-folder-option")
    ).find((r) => r.textContent?.includes("Archive"));
    fireEvent.click(archive as HTMLElement);

    await waitFor(() =>
      expect(
        screen.getByTestId("confirm-move-to-folder").hasAttribute("disabled"),
      ).toBe(false),
    );
  });

  test("a name clash keeps the move dialog open and names the folder", async () => {
    useNestedHandlers();
    server.use(
      http.patch(`${PROXY}/folders/:folderId`, () =>
        HttpResponse.json({ detail: "exists" }, { status: 409 }),
      ),
    );

    render(<ArtifactsPage />);
    await openFolderMenu("Reports");
    fireEvent.click(await screen.findByTestId("folder-move-menu"));
    const archive = (
      await screen.findAllByTestId("move-to-folder-option")
    ).find((r) => r.textContent?.includes("Archive"));
    fireEvent.click(archive as HTMLElement);
    fireEvent.click(screen.getByTestId("confirm-move-to-folder"));

    // The dialog survives the clash so another destination can be picked
    // without reopening it; the message itself is unit-tested on
    // describeFolderMoveError.
    await waitFor(() =>
      expect(
        screen.getByTestId("confirm-move-to-folder").hasAttribute("disabled"),
      ).toBe(false),
    );
    expect(
      screen.getAllByTestId("move-to-folder-option").length,
    ).toBeGreaterThan(0);
  });

  test("the tree moves focus with the arrow keys and selects with Enter", async () => {
    useNestedHandlers();

    render(<ArtifactsPage />);
    await openFolderMenu("Reports");
    fireEvent.click(await screen.findByTestId("folder-move-menu"));

    // Rows are Archive, Reports (the subject), 2026 — sorted by name, with
    // the subject's own location already expanded.
    const rows = await screen.findAllByTestId("move-to-folder-option");
    const reports = rows.findIndex((r) => r.textContent?.includes("Reports"));
    rows[reports].focus();
    fireEvent.keyDown(rows[reports], { key: "Enter" });

    // "Reports" is the folder being moved: refused, so Enter selects nothing.
    expect(
      screen.getByTestId("confirm-move-to-folder").hasAttribute("disabled"),
    ).toBe(true);

    const archive = rows.findIndex((r) => r.textContent?.includes("Archive"));
    fireEvent.keyDown(rows[reports], {
      key: archive < reports ? "ArrowUp" : "ArrowDown",
    });
    expect(document.activeElement).toBe(rows[archive]);
    fireEvent.keyDown(rows[archive], { key: "Enter" });
    await waitFor(() =>
      expect(
        screen.getByTestId("confirm-move-to-folder").hasAttribute("disabled"),
      ).toBe(false),
    );
  });
});
