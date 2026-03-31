import { renderHook, waitFor } from "@testing-library/react";
import { UIDataTypes, UIMessage, UITools } from "ai";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useCopilotUIStore } from "../../store";
import { useAutoOpenArtifacts } from "./useAutoOpenArtifacts";

function createMessage(
  id: string,
  role: UIMessage<unknown, UIDataTypes, UITools>["role"],
  text: string,
): UIMessage<unknown, UIDataTypes, UITools> {
  return {
    id,
    role,
    parts: [{ type: "text", text }],
  };
}

describe("useAutoOpenArtifacts", () => {
  const mockOpenArtifact = vi.fn();

  beforeEach(() => {
    mockOpenArtifact.mockReset();
    useCopilotUIStore.setState({
      openArtifact: mockOpenArtifact,
      artifactPanel: {
        isOpen: false,
        isMinimized: false,
        isMaximized: false,
        width: 600,
        activeArtifact: null,
        history: [],
      },
    });
  });

  it("does not auto-open artifacts from the initial message hydration", () => {
    renderHook(() =>
      useAutoOpenArtifacts({
        sessionId: "session-1",
        messages: [
          createMessage(
            "assistant-1",
            "assistant",
            "[Quarterly Report](workspace://550e8400-e29b-41d4-a716-446655440000#application/pdf)",
          ),
        ],
      }),
    );

    expect(mockOpenArtifact).not.toHaveBeenCalled();
  });

  it("auto-opens an artifact from a new assistant message", async () => {
    const { rerender } = renderHook(
      ({
        messages,
        sessionId,
      }: {
        messages: UIMessage<unknown, UIDataTypes, UITools>[];
        sessionId: string;
      }) => useAutoOpenArtifacts({ messages, sessionId }),
      {
        initialProps: {
          sessionId: "session-1",
          messages: [createMessage("user-1", "user", "Build the report")],
        },
      },
    );

    rerender({
      sessionId: "session-1",
      messages: [
        createMessage("user-1", "user", "Build the report"),
        createMessage(
          "assistant-1",
          "assistant",
          "[Executive Dashboard](workspace://550e8400-e29b-41d4-a716-446655440001#text/html)",
        ),
      ],
    });

    await waitFor(() => {
      expect(mockOpenArtifact).toHaveBeenCalledWith({
        id: "550e8400-e29b-41d4-a716-446655440001",
        title: "Executive Dashboard",
        mimeType: "text/html",
        origin: "agent",
        sourceUrl:
          "/api/proxy/api/workspace/files/550e8400-e29b-41d4-a716-446655440001/download",
      });
    });
  });

  it("auto-opens an artifact when an assistant message is edited to include one", async () => {
    const { rerender } = renderHook(
      ({ messageText }: { messageText: string }) =>
        useAutoOpenArtifacts({
          sessionId: "session-1",
          messages: [createMessage("assistant-1", "assistant", messageText)],
        }),
      {
        initialProps: {
          messageText: "Working on it now.",
        },
      },
    );

    rerender({
      messageText:
        "Working on it now.\n\n[Notification Toast](workspace://550e8400-e29b-41d4-a716-446655440002#text/jsx)",
    });

    await waitFor(() => {
      expect(mockOpenArtifact).toHaveBeenCalledWith({
        id: "550e8400-e29b-41d4-a716-446655440002",
        title: "Notification Toast",
        mimeType: "text/jsx",
        origin: "agent",
        sourceUrl:
          "/api/proxy/api/workspace/files/550e8400-e29b-41d4-a716-446655440002/download",
      });
    });
  });
});
