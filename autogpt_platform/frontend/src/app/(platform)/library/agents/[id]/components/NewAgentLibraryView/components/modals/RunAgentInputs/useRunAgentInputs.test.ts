import { act, renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useOrgTeamStore } from "@/services/org-team/store";
import {
  ORG_HEADER_NAME,
  TEAM_HEADER_NAME,
} from "@/services/org-team/header-names";
import { useRunAgentInputs } from "./useRunAgentInputs";

const mocks = vi.hoisted(() => ({
  upload: vi.fn(),
}));

vi.mock("@/app/api/__generated__/endpoints/files/files", () => ({
  postV1UploadFileToCloudStorage: mocks.upload,
}));

describe("useRunAgentInputs", () => {
  beforeEach(() => {
    mocks.upload.mockReset();
    mocks.upload.mockResolvedValue({
      status: 200,
      data: {
        file_name: "example.txt",
        size: 7,
        content_type: "text/plain",
        file_url: "https://example.test/file",
      },
      headers: new Headers(),
    });
    useOrgTeamStore.setState({
      activeOrgID: "org-nav",
      activeTeamID: "team-nav",
      isLoaded: true,
    });
  });

  it("uploads run inputs into the resource scope instead of the navbar scope", async () => {
    const { result } = renderHook(() =>
      useRunAgentInputs({
        organizationId: "org-agent",
        teamId: "team-agent",
      }),
    );
    const file = new File(["content"], "example.txt", {
      type: "text/plain",
    });

    await act(async () => {
      await result.current.handleUploadFile(file);
    });

    const request = mocks.upload.mock.calls[0][2] as RequestInit;
    const headers = new Headers(request.headers);
    expect(headers.get(ORG_HEADER_NAME)).toBe("org-agent");
    expect(headers.get(TEAM_HEADER_NAME)).toBe("team-agent");
  });
});
