import { getListWorkspaceFilesMockHandler200 } from "@/app/api/__generated__/endpoints/workspace/workspace.msw";
import type { ListFilesResponse } from "@/app/api/__generated__/models/listFilesResponse";
import { server } from "@/mocks/mock-server";
import { fireEvent, render, screen } from "@/tests/integrations/test-utils";
import { afterEach, describe, expect, it } from "vitest";
import { useCopilotUIStore } from "../../../store";
import { ArtifactsTab } from "../components/ArtifactsTab/ArtifactsTab";

const SESSION = "session-1";

function listResponse(names: string[]): ListFilesResponse {
  return {
    files: names.map((name, i) => ({
      id: `cccccccc-0000-0000-0000-00000000000${i}`,
      name,
      path: `/sessions/${SESSION}/${name}`,
      mime_type: "text/markdown",
      size_bytes: 2048,
      metadata: { origin: "agent" },
      origin: "generated" as const,
      created_at: "2026-05-20T10:00:00Z",
    })),
    offset: 0,
    has_more: false,
  };
}

afterEach(() => {
  useCopilotUIStore.setState({
    artifactPanel: {
      isOpen: false,
      activeArtifact: null,
      history: [],
      activeTab: "files",
      lastArtifact: null,
    },
  });
});

describe("ArtifactsTab", () => {
  it("previews this chat's artifacts with a link to the library", async () => {
    server.use(getListWorkspaceFilesMockHandler200(listResponse(["notes.md"])));
    render(<ArtifactsTab sessionId={SESSION} />);

    expect(await screen.findByText("notes.md")).toBeDefined();
    expect(screen.getByText(/Pick an artifact from this chat/i)).toBeDefined();
    expect(
      screen
        .getByRole("link", { name: /Open artifacts/i })
        .getAttribute("href"),
    ).toBe("/artifacts");
  });

  it("shows the empty state when the chat produced nothing", async () => {
    server.use(getListWorkspaceFilesMockHandler200(listResponse([])));
    render(<ArtifactsTab sessionId={SESSION} />);

    expect(await screen.findByText(/Nothing to preview yet/i)).toBeDefined();
  });

  it("opens a mini card's artifact in the preview panel", async () => {
    server.use(getListWorkspaceFilesMockHandler200(listResponse(["notes.md"])));
    render(<ArtifactsTab sessionId={SESSION} />);

    fireEvent.click(await screen.findByText("notes.md"));

    expect(
      useCopilotUIStore.getState().artifactPanel.activeArtifact?.title,
    ).toBe("notes.md");
  });

  it("caps the preview list at six artifacts", async () => {
    server.use(
      getListWorkspaceFilesMockHandler200(
        listResponse(Array.from({ length: 8 }, (_, i) => `file-${i}.md`)),
      ),
    );
    render(<ArtifactsTab sessionId={SESSION} />);

    expect(await screen.findByText("file-0.md")).toBeDefined();
    expect(screen.queryByText("file-6.md")).toBeNull();
  });
});
