import { act, renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { buildWorkspaceURI } from "@/lib/workspace-uri";
import { useOrgTeamStore } from "@/services/org-team/store";
import {
  ORG_HEADER_NAME,
  TEAM_HEADER_NAME,
} from "@/services/org-team/header-names";
import { useWorkspaceUpload } from "./useWorkspaceUpload";

const mocks = vi.hoisted(() => ({
  deleteFile: vi.fn(),
  deleteRequest: undefined as RequestInit | undefined,
  uploadFileDirect: vi.fn(),
}));

vi.mock("@/app/api/__generated__/endpoints/workspace/workspace", () => ({
  useDeleteWorkspaceFile: vi.fn(
    (options: { request?: RequestInit } | undefined) => {
      mocks.deleteRequest = options?.request;
      return { mutate: mocks.deleteFile };
    },
  ),
}));

vi.mock("@/lib/direct-upload", () => ({
  uploadFileDirect: mocks.uploadFileDirect,
}));

describe("useWorkspaceUpload", () => {
  beforeEach(() => {
    mocks.deleteFile.mockClear();
    mocks.uploadFileDirect.mockReset();
    mocks.deleteRequest = undefined;
    mocks.uploadFileDirect.mockResolvedValue({
      file_id: "file-a",
      name: "example.txt",
      mime_type: "text/plain",
      size_bytes: 7,
    });
    useOrgTeamStore.setState({
      activeOrgID: "org-nav",
      activeTeamID: "team-nav",
      isLoaded: true,
    });
  });

  it("uses the builder resource scope instead of the navbar for upload and delete", async () => {
    const { result } = renderHook(() =>
      useWorkspaceUpload({
        organizationId: "org-builder",
        teamId: "team-builder",
      }),
    );
    const file = new File(["content"], "example.txt", {
      type: "text/plain",
    });

    await act(async () => {
      await result.current.handleUploadFile(file);
    });
    act(() => {
      result.current.handleDeleteFile(
        buildWorkspaceURI("file-a", "text/plain"),
      );
    });

    expect(mocks.uploadFileDirect).toHaveBeenCalledWith(file, undefined, {
      organizationId: "org-builder",
      teamId: "team-builder",
    });
    const headers = new Headers(mocks.deleteRequest?.headers);
    expect(headers.get(ORG_HEADER_NAME)).toBe("org-builder");
    expect(headers.get(TEAM_HEADER_NAME)).toBe("team-builder");
    expect(mocks.deleteFile).toHaveBeenCalledWith({ fileId: "file-a" });
  });
});
