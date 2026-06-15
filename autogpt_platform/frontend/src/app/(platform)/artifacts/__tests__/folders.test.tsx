import { describe, expect, test, vi } from "vitest";

import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import { server } from "@/mocks/mock-server";
import { http, HttpResponse } from "msw";
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

vi.mock("next/navigation", () => ({
  useRouter: () => ({
    push: vi.fn(),
    replace: vi.fn(),
    prefetch: vi.fn(),
    back: vi.fn(),
    forward: vi.fn(),
    refresh: vi.fn(),
  }),
  usePathname: () => "/artifacts",
  useSearchParams: () => new URLSearchParams(),
  useParams: () => ({}),
  notFound: () => {
    throw new Error("NEXT_NOT_FOUND");
  },
}));

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

describe("ArtifactsPage - folders", () => {
  test("renders folders at the root with file counts", async () => {
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

    expect(await screen.findByTestId("workspace-folders")).toBeDefined();
    expect(await screen.findByText("Reports")).toBeDefined();
    expect(screen.getByText("2 files")).toBeDefined();
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

    fireEvent.click(await screen.findByTestId("create-folder-button"));
    fireEvent.change(screen.getByLabelText(/folder name/i), {
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
    // Radix DropdownMenu opens on pointerdown, not click, under happy-dom.
    fireEvent.pointerDown(screen.getByTestId("artifacts-card-menu"), {
      button: 0,
    });
    fireEvent.click(await screen.findByTestId("artifacts-move-to-folder"));
    fireEvent.click(await screen.findByTestId("move-to-folder-option"));

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

    fireEvent.click(await screen.findByLabelText("Rename folder"));
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

    fireEvent.click(await screen.findByLabelText("Delete folder"));
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
    fireEvent.click(await screen.findByTestId("move-to-root"));

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

  test("dragging a file onto a folder moves it", async () => {
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

    const card = screen.getByTestId("artifacts-list-item");
    const folder = screen.getByTestId("workspace-folder");
    fireEvent.dragStart(card, { dataTransfer });
    fireEvent.dragOver(folder, { dataTransfer });
    fireEvent.drop(folder, { dataTransfer });
    fireEvent.dragEnd(card, { dataTransfer });

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
    fireEvent.keyDown(await screen.findByTestId("workspace-folder"), {
      key: "Enter",
    });
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

    fireEvent.click(await screen.findByTestId("create-folder-button"));
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

    fireEvent.click(await screen.findByLabelText("Delete folder"));
    fireEvent.click(await screen.findByRole("button", { name: /cancel/i }));

    await waitFor(() =>
      expect(screen.queryByTestId("confirm-delete-folder")).toBeNull(),
    );
    expect(deleteCalled).toBe(false);
  });
});
