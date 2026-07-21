import { beforeEach, describe, expect, test } from "vitest";
import { render, waitFor } from "@/tests/integrations/test-utils";
import { server } from "@/mocks/mock-server";
import { getListWorkspaceFilesMockHandler200 } from "@/app/api/__generated__/endpoints/workspace/workspace.msw";
import type { ListFilesResponse } from "@/app/api/__generated__/models/listFilesResponse";
import { useCopilotUIStore } from "../../../store";
import { ContextPanelAutoOpen } from "../ContextPanelAutoOpen";

const SESSION = "session-1";

function withFiles(): ListFilesResponse {
  return {
    files: [
      {
        id: "aaaaaaaa-0000-0000-0000-000000000001",
        name: "result.csv",
        path: "/sessions/session-1/result.csv",
        mime_type: "text/csv",
        size_bytes: 4096,
        metadata: { origin: "agent" },
        origin: "generated",
        created_at: "2026-05-20T11:00:00Z",
      },
    ],
    offset: 0,
    has_more: false,
  };
}

describe("ContextPanelAutoOpen", () => {
  beforeEach(() => {
    useCopilotUIStore.getState().resetAutoOpenState?.();
    useCopilotUIStore.setState((s) => ({
      artifactPanel: {
        ...s.artifactPanel,
        isOpen: false,
        activeArtifact: null,
      },
    }));
  });

  test("opens the last generated file in the artifact panel when the session has files", async () => {
    server.use(getListWorkspaceFilesMockHandler200(withFiles()));
    render(<ContextPanelAutoOpen sessionId={SESSION} />);
    await waitFor(() =>
      expect(
        useCopilotUIStore.getState().artifactPanel.activeArtifact?.id,
      ).toBe("aaaaaaaa-0000-0000-0000-000000000001"),
    );
    expect(useCopilotUIStore.getState().artifactPanel.isOpen).toBe(true);
  });
});
