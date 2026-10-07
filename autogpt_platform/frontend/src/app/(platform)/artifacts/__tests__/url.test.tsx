import { beforeEach, describe, expect, test, vi } from "vitest";

import {
  fireEvent,
  render,
  screen,
  waitFor,
} from "@/tests/integrations/test-utils";
import { server } from "@/mocks/mock-server";
import { http, HttpResponse } from "msw";
import { getListExpertIdentitiesMockHandler } from "@/app/api/__generated__/endpoints/experts/experts.msw";
import { getGetWorkspaceStorageUsageMockHandler } from "@/app/api/__generated__/endpoints/workspace/workspace.msw";
import {
  currentFolderParam,
  resetNavigation,
  routerMock,
} from "./navigation-mock";

vi.mock("@/services/feature-flags/use-get-flag", () => ({
  Flag: { ARTIFACTS_PAGE: "artifacts-page" },
  useGetFlag: () => true,
  useFlagStatus: () => ({ enabled: true, ready: true }),
}));

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

// Root lists root.txt, the one live folder lists inside.txt — so which listing
// rendered says which folder the page opened.
function useBaseHandlers(
  experts: Parameters<typeof getListExpertIdentitiesMockHandler>[0] = [],
) {
  server.use(
    getListExpertIdentitiesMockHandler(experts),
    getGetWorkspaceStorageUsageMockHandler({
      used_bytes: 0,
      limit_bytes: 1_000_000_000,
      used_percent: 0,
      file_count: 0,
    }),
    http.get(`${PROXY}/folders`, () =>
      HttpResponse.json({
        folders: [
          {
            id: "fld-1",
            workspace_id: "ws-1",
            name: "Reports",
            file_count: 1,
            created_at: "2026-05-01T00:00:00Z",
            updated_at: "2026-05-01T00:00:00Z",
          },
        ],
      }),
    ),
    http.get(`${PROXY}/files`, ({ request }) => {
      const folderId = new URL(request.url).searchParams.get("folder_id");
      return HttpResponse.json({
        files: [
          folderId === "fld-1"
            ? {
                ...baseFile,
                id: "f-in",
                name: "inside.txt",
                folder_id: "fld-1",
              }
            : { ...baseFile, id: "f-root", name: "root.txt" },
        ],
        offset: 0,
        has_more: false,
      });
    }),
  );
}

const baseFile = {
  id: "file-base",
  name: "base.txt",
  path: "/base.txt",
  mime_type: "text/plain",
  size_bytes: 1024,
  folder_id: null,
  metadata: {},
  origin: "generated",
  created_at: "2026-05-01T00:00:00Z",
};

describe("ArtifactsPage - folder in the URL", () => {
  beforeEach(() => resetNavigation());

  test("opens the folder named by the URL on load", async () => {
    resetNavigation("folder=fld-1");
    useBaseHandlers();

    render(<ArtifactsPage />);

    expect(await screen.findByText("inside.txt")).toBeDefined();
    expect(await screen.findByTestId("folder-breadcrumb")).toBeDefined();
    expect(screen.getByText("Reports")).toBeDefined();
    // The root listing must never paint on the way in.
    expect(screen.queryByText("root.txt")).toBeNull();
  });

  test("selecting a folder pushes it into the URL", async () => {
    useBaseHandlers();

    render(<ArtifactsPage />);
    expect(await screen.findByText("root.txt")).toBeDefined();

    fireEvent.click(await screen.findByTestId("workspace-folder"));

    await waitFor(() => expect(currentFolderParam()).toBe("fld-1"));
    expect(routerMock.push).toHaveBeenCalledWith("/artifacts?folder=fld-1");
    expect(await screen.findByText("inside.txt")).toBeDefined();
  });

  test("going back to root replaces the folder out of the URL", async () => {
    resetNavigation("folder=fld-1");
    useBaseHandlers();

    render(<ArtifactsPage />);
    await screen.findByTestId("folder-breadcrumb");

    fireEvent.click(screen.getByTestId("folder-breadcrumb-root"));

    await waitFor(() => expect(currentFolderParam()).toBeNull());
    expect(routerMock.replace).toHaveBeenCalledWith("/artifacts");
    expect(await screen.findByText("root.txt")).toBeDefined();
  });

  test("a folder id that no longer exists falls back to root", async () => {
    resetNavigation("folder=deleted-folder");
    useBaseHandlers();

    render(<ArtifactsPage />);

    expect(await screen.findByText("root.txt")).toBeDefined();
    expect(screen.queryByTestId("folder-breadcrumb")).toBeNull();
    // The dead id is dropped, so re-sharing the URL does not carry it on.
    await waitFor(() => expect(currentFolderParam()).toBeNull());
  });

  test("filtering by an expert narrows inside the open folder instead of leaving it", async () => {
    resetNavigation("folder=fld-1");
    const requests: URLSearchParams[] = [];
    useBaseHandlers([
      {
        id: "expert-a",
        name: "Maria",
        avatar_url: null,
        role: "Analyst",
        is_archived: false,
      },
    ]);
    server.use(
      http.get(`${PROXY}/files`, ({ request }) => {
        requests.push(new URL(request.url).searchParams);
        return HttpResponse.json({
          files: [{ ...baseFile, id: "f-in", name: "inside.txt" }],
          offset: 0,
          has_more: false,
        });
      }),
    );

    render(<ArtifactsPage />);
    await screen.findByTestId("folder-breadcrumb");

    fireEvent.click(
      await screen.findByTestId("artifacts-expert-filter-expert-a"),
    );

    await waitFor(() =>
      expect(
        requests.some(
          (params) =>
            params.get("expert_id") === "expert-a" &&
            params.get("folder_id") === "fld-1",
        ),
      ).toBe(true),
    );
    expect(currentFolderParam()).toBe("fld-1");
    expect(screen.getByTestId("folder-breadcrumb")).toBeDefined();
    // "From: Maria" keeps meaning "made in Maria's chats": widening to the
    // user's own uploads is the picker's switch, never this tab.
    expect(
      requests.every((params) => params.get("include_user_files") === null),
    ).toBe(true);
  });

  test("at the root an expert filter still asks for root-level files only", async () => {
    const requests: URLSearchParams[] = [];
    useBaseHandlers([
      {
        id: "expert-a",
        name: "Maria",
        avatar_url: null,
        role: "Analyst",
        is_archived: false,
      },
    ]);
    server.use(
      http.get(`${PROXY}/files`, ({ request }) => {
        requests.push(new URL(request.url).searchParams);
        return HttpResponse.json({
          files: [{ ...baseFile, id: "f-root", name: "root.txt" }],
          offset: 0,
          has_more: false,
        });
      }),
    );

    render(<ArtifactsPage />);
    expect(await screen.findByText("root.txt")).toBeDefined();

    fireEvent.click(
      await screen.findByTestId("artifacts-expert-filter-expert-a"),
    );

    await waitFor(() =>
      expect(
        requests.some(
          (params) =>
            params.get("expert_id") === "expert-a" &&
            params.get("root_only") === "true",
        ),
      ).toBe(true),
    );
  });

  test("an ancestor crumb pushes that ancestor onto the URL", async () => {
    resetNavigation("folder=fld-3");
    useBaseHandlers();
    server.use(
      http.get(`${PROXY}/folders`, () =>
        HttpResponse.json({ folders: NESTED_FOLDERS }),
      ),
    );

    render(<ArtifactsPage />);
    // The crumb paints with a placeholder before the folders query settles,
    // so wait for the open folder's own name rather than the nav.
    await screen.findByText("Q3");

    fireEvent.click(await screen.findByRole("button", { name: "Reports" }));

    // A push, not a replace: back returns to the folder you jumped from.
    await waitFor(() => expect(currentFolderParam()).toBe("fld-1"));
    expect(routerMock.push).toHaveBeenCalled();
  });

  test("more than four crumbs collapse into a menu of the hidden ancestors", async () => {
    resetNavigation("folder=fld-4");
    useBaseHandlers();
    server.use(
      http.get(`${PROXY}/folders`, () =>
        HttpResponse.json({ folders: NESTED_FOLDERS }),
      ),
    );

    render(<ArtifactsPage />);
    await screen.findByText("Week 1");

    // Root, the parent and the current folder stay; "Reports" hides.
    expect(screen.queryByRole("button", { name: "Reports" })).toBeNull();
    fireEvent.pointerDown(screen.getByTestId("folder-breadcrumb-overflow"), {
      button: 0,
    });
    fireEvent.click(await screen.findByRole("menuitem", { name: "Reports" }));

    await waitFor(() => expect(currentFolderParam()).toBe("fld-1"));
  });
});

// Files / Reports / 2026 / Q3 / Week 1
const NESTED_FOLDERS = [
  { id: "fld-1", name: "Reports", parent_id: null },
  { id: "fld-2", name: "2026", parent_id: "fld-1" },
  { id: "fld-3", name: "Q3", parent_id: "fld-2" },
  { id: "fld-4", name: "Week 1", parent_id: "fld-3" },
].map((folder) => ({
  ...folder,
  workspace_id: "ws-1",
  file_count: 0,
  created_at: "2026-05-01T00:00:00Z",
  updated_at: "2026-05-01T00:00:00Z",
}));
