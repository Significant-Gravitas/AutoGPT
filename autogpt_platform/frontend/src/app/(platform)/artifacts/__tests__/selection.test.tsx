import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";

import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import { server } from "@/mocks/mock-server";
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

beforeEach(() => {
  server.use(getListExpertIdentitiesMockHandler([]));
});

afterEach(() => {
  vi.restoreAllMocks();
});

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

function useBaseHandlers(files: WorkspaceFileItem[]) {
  server.use(
    getGetWorkspaceStorageUsageMockHandler({
      used_bytes: 0,
      limit_bytes: 1_000_000_000,
      used_percent: 0,
      file_count: 0,
    }),
    getListWorkspaceFilesMockHandler({ files, offset: 0, has_more: false }),
    getListWorkspaceFoldersMockHandler({
      folders: [makeFolder({ id: "fld-1", name: "Reports" })],
    }),
  );
}

function makeDataTransfer() {
  const store: Record<string, string> = {};
  return {
    store,
    dataTransfer: {
      setData: (key: string, value: string) => {
        store[key] = value;
      },
      getData: (key: string) => store[key] ?? "",
      setDragImage: () => {},
      get types() {
        return Object.keys(store);
      },
    },
  };
}

const TWO_FILES = [
  makeFile({ id: "f1", name: "one.txt" }),
  makeFile({ id: "f2", name: "two.txt" }),
];

describe("ArtifactsPage - row selection", () => {
  test("clicking a thumbnail selects the row without opening it", async () => {
    useBaseHandlers(TWO_FILES);

    render(<ArtifactsPage />);

    const toggle = await screen.findByLabelText("Select one.txt");
    fireEvent.click(toggle);

    expect(await screen.findByTestId("artifacts-selection-bar")).toBeDefined();
    expect(screen.getByText("1 selected")).toBeDefined();
    expect(screen.queryByTestId("file-viewer")).toBeNull();
    expect(toggle.getAttribute("aria-pressed")).toBe("true");

    fireEvent.click(toggle);
    await waitFor(() =>
      expect(screen.queryByTestId("artifacts-selection-bar")).toBeNull(),
    );
    expect(screen.getByText("Modified")).toBeDefined();
  });

  test("select all and clear work from the selection bar", async () => {
    useBaseHandlers(TWO_FILES);

    render(<ArtifactsPage />);

    fireEvent.click(await screen.findByLabelText("Select one.txt"));
    fireEvent.click(await screen.findByTestId("artifacts-select-all"));

    expect(await screen.findByText("2 selected")).toBeDefined();
    expect(screen.queryByTestId("artifacts-select-all")).toBeNull();

    fireEvent.click(screen.getByLabelText("Clear selection"));
    await waitFor(() =>
      expect(screen.queryByTestId("artifacts-selection-bar")).toBeNull(),
    );
  });

  test("deleting the selection deletes every selected file", async () => {
    useBaseHandlers(TWO_FILES);
    const deleted: string[] = [];
    server.use(
      http.delete(`${PROXY}/files/:fileId`, ({ params }) => {
        deleted.push(String(params.fileId));
        return new HttpResponse(null, { status: 204 });
      }),
    );
    vi.spyOn(window, "confirm").mockReturnValue(true);

    render(<ArtifactsPage />);

    fireEvent.click(await screen.findByLabelText("Select one.txt"));
    fireEvent.click(await screen.findByLabelText("Select two.txt"));
    fireEvent.click(await screen.findByTestId("artifacts-selection-delete"));

    await waitFor(() => expect(deleted.sort()).toEqual(["f1", "f2"]));
  });

  test("a cancelled delete confirmation deletes nothing", async () => {
    useBaseHandlers(TWO_FILES);
    let deleteCalled = false;
    server.use(
      http.delete(`${PROXY}/files/:fileId`, () => {
        deleteCalled = true;
        return new HttpResponse(null, { status: 204 });
      }),
    );
    vi.spyOn(window, "confirm").mockReturnValue(false);

    render(<ArtifactsPage />);

    fireEvent.click(await screen.findByLabelText("Select one.txt"));
    fireEvent.click(await screen.findByTestId("artifacts-selection-delete"));

    expect(await screen.findByText("1 selected")).toBeDefined();
    expect(deleteCalled).toBe(false);
  });

  test("moving the selection posts one bulk move with every id", async () => {
    useBaseHandlers(TWO_FILES);
    let body: { file_ids: string[]; folder_id: string | null } | null = null;
    server.use(
      http.post(`${PROXY}/folders/files/bulk-move`, async ({ request }) => {
        body = (await request.json()) as typeof body;
        return HttpResponse.json([]);
      }),
    );

    render(<ArtifactsPage />);

    fireEvent.click(await screen.findByLabelText("Select one.txt"));
    fireEvent.click(await screen.findByLabelText("Select two.txt"));
    fireEvent.click(await screen.findByTestId("artifacts-selection-move"));
    expect(await screen.findByText("Move 2 files to:")).toBeDefined();
    fireEvent.click(await screen.findByTestId("move-to-folder-option"));

    await waitFor(() =>
      expect(body).toEqual({ file_ids: ["f1", "f2"], folder_id: "fld-1" }),
    );
  });

  test("dragging a selected row onto a folder moves the whole selection", async () => {
    useBaseHandlers(TWO_FILES);
    let body: { file_ids: string[]; folder_id: string | null } | null = null;
    server.use(
      http.post(`${PROXY}/folders/files/bulk-move`, async ({ request }) => {
        body = (await request.json()) as typeof body;
        return HttpResponse.json([]);
      }),
    );

    render(<ArtifactsPage />);

    fireEvent.click(await screen.findByLabelText("Select one.txt"));
    fireEvent.click(await screen.findByLabelText("Select two.txt"));

    const { store, dataTransfer } = makeDataTransfer();
    const [firstRow] = screen.getAllByTestId("artifacts-list-item");
    const folder = await screen.findByTestId("workspace-folder");
    fireEvent.dragStart(firstRow, { dataTransfer });
    fireEvent.dragOver(folder, { dataTransfer });
    fireEvent.drop(folder, { dataTransfer });
    fireEvent.dragEnd(firstRow, { dataTransfer });

    expect(store[FILE_DRAG_MIME]).toBe("f1,f2");
    await waitFor(() =>
      expect(body).toEqual({ file_ids: ["f1", "f2"], folder_id: "fld-1" }),
    );
  });

  test("dragging an unselected row carries only that file", async () => {
    useBaseHandlers(TWO_FILES);
    server.use(
      http.post(`${PROXY}/folders/files/bulk-move`, () =>
        HttpResponse.json([]),
      ),
    );

    render(<ArtifactsPage />);

    fireEvent.click(await screen.findByLabelText("Select one.txt"));

    const { store, dataTransfer } = makeDataTransfer();
    const [, secondRow] = screen.getAllByTestId("artifacts-list-item");
    fireEvent.dragStart(secondRow, { dataTransfer });
    fireEvent.dragEnd(secondRow, { dataTransfer });

    expect(store[FILE_DRAG_MIME]).toBe("f2");
  });
});
