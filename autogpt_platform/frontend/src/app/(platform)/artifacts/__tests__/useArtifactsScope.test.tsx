import { act, renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import type { WorkspaceFolder } from "@/app/api/__generated__/models/workspaceFolder";
import { useOrgTeamStore } from "@/services/org-team/store";

const { flag, listFiles, listFolders } = vi.hoisted(() => ({
  flag: { enabled: true },
  listFiles: vi.fn(),
  listFolders: vi.fn(),
}));

vi.mock("@/services/feature-flags/use-get-flag", () => ({
  Flag: { SHOW_ORG_SETTINGS: "SHOW_ORG_SETTINGS" },
  useGetFlag: () => flag.enabled,
}));
vi.mock("@/app/api/__generated__/endpoints/workspace/workspace", () => ({
  listWorkspaceFiles: listFiles,
  useListWorkspaceFolders: listFolders,
  getListWorkspaceFoldersQueryKey: () => ["/workspace/folders"],
  bulkMoveWorkspaceFiles: vi.fn(),
  createWorkspaceFolder: vi.fn(),
  deleteWorkspaceFolder: vi.fn(),
  updateWorkspaceFolder: vi.fn(),
}));
vi.mock("@/components/molecules/Toast/use-toast", () => ({
  useToast: () => ({ toast: vi.fn() }),
}));

import { useArtifactsPage } from "../useArtifactsPage";
import { useArtifactsFolders } from "../useArtifactsFolders";

let queryClient: QueryClient;

function wrapper({ children }: { children: React.ReactNode }) {
  return (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
}

function folder(orgId: string, teamId: string): WorkspaceFolder {
  return {
    id: `${teamId}-folder`,
    name: "Folder",
    organization_id: orgId,
    team_id: teamId,
  } as WorkspaceFolder;
}

beforeEach(() => {
  flag.enabled = true;
  queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  useOrgTeamStore.setState({
    activeOrgID: "personal",
    activeTeamID: "default-team",
    isLoaded: true,
  });
  listFiles
    .mockReset()
    .mockResolvedValue({ status: 200, data: { files: [], has_more: false } });
  listFolders.mockReset().mockReturnValue({ data: { folders: [] } });
});

afterEach(() => {
  queryClient.clear();
  useOrgTeamStore.getState().clearContext();
});

it("clears a selected shared folder and resumes personal listing when collaboration turns off", async () => {
  const { result, rerender } = renderHook(() => useArtifactsPage(), {
    wrapper,
  });
  await waitFor(() => expect(listFiles).toHaveBeenCalled());
  act(() => result.current.selectFolder(folder("shared", "shared-team")));
  await waitFor(() =>
    expect(listFiles).toHaveBeenCalledWith(
      expect.objectContaining({ folder_id: "shared-team-folder" }),
      expect.objectContaining({
        headers: expect.objectContaining({
          "X-Org-Id": "shared",
          "X-Team-Id": "shared-team",
        }),
      }),
    ),
  );
  listFiles.mockClear();

  flag.enabled = false;
  rerender();

  expect(result.current.selectedFolderId).toBeNull();
  await waitFor(() => expect(listFiles).toHaveBeenCalled());
  for (const [params, options] of listFiles.mock.calls) {
    expect(params.folder_id).toBeUndefined();
    expect(options.headers).toMatchObject({
      "X-Org-Id": "personal",
      "X-Team-Id": "default-team",
    });
  }
});

it("rejects foreign folder selection while disabled but preserves personal folders", async () => {
  flag.enabled = false;
  const { result } = renderHook(() => useArtifactsPage(), { wrapper });
  act(() => result.current.selectFolder(folder("shared", "shared-team")));
  expect(result.current.selectedFolderId).toBeNull();
  act(() => result.current.selectFolder(folder("personal", "default-team")));
  expect(result.current.selectedFolderId).toBe("default-team-folder");
});

it("uses custom folder-list scope only while collaboration is enabled", () => {
  const { rerender } = renderHook(
    () =>
      useArtifactsFolders({
        organizationId: "shared",
        teamId: "shared-team",
      }),
    { wrapper },
  );
  expect(listFolders.mock.lastCall?.[0].request.headers).toMatchObject({
    "X-Org-Id": "shared",
    "X-Team-Id": "shared-team",
  });

  flag.enabled = false;
  rerender();

  expect(listFolders.mock.lastCall?.[0].request.headers).toMatchObject({
    "X-Org-Id": "personal",
    "X-Team-Id": "default-team",
  });
});
