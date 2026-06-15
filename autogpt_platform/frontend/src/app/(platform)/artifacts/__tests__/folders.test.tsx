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
});
