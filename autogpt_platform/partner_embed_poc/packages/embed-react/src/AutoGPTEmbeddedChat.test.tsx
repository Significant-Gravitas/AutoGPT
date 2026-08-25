import type { UIMessage } from "ai";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { AutoGPTEmbeddedChat } from "./AutoGPTEmbeddedChat";
const sessionChatMock = vi.hoisted(() => ({ messages: [] as UIMessage[] }));

vi.mock("./api", () => ({
  createEmbedSession: vi.fn().mockResolvedValue({
    id: "new-session",
    createdAt: "2026-08-24T12:00:00Z",
  }),
}));

vi.mock("./session-api", () => ({
  listEmbedSessions: vi.fn().mockResolvedValue([
    {
      id: "session-1",
      title: "Daily operations",
      createdAt: "2026-08-24T10:00:00Z",
      updatedAt: "2026-08-24T11:00:00Z",
      chatStatus: "idle",
    },
  ]),
  getEmbedSession: vi.fn().mockResolvedValue({
    id: "session-1",
    title: "Daily operations",
    createdAt: "2026-08-24T10:00:00Z",
    updatedAt: "2026-08-24T11:00:00Z",
    chatStatus: "idle",
    messages: [],
    hasMoreMessages: false,
    oldestSequence: null,
    capabilities: [
      "documents.read",
      "autogpt:block:b1ab9b19-67a6-406d-abf5-2dba76d00c79",
    ],
  }),
  listEmbedArtifacts: vi.fn().mockResolvedValue([
    {
      id: "artifact-1",
      name: "exceptions.csv",
      path: "/sessions/session-1/exceptions.csv",
      mimeType: "text/csv",
      sizeBytes: 1024,
      createdAt: "2026-08-24T11:00:00Z",
    },
  ]),
  downloadEmbedArtifact: vi.fn(),
}));

vi.mock("./useSessionChat", () => ({
  useSessionChat: () => ({
    messages: sessionChatMock.messages,
    sendMessage: vi.fn(),
    stop: vi.fn(),
    status: "ready",
    error: null,
  }),
}));

describe("AutoGPTEmbeddedChat", () => {
  beforeEach(() => {
    cleanup();
    sessionChatMock.messages = [];
  });

  it("restores native chat surfaces without asking for AutoGPT credentials", async () => {
    render(
      <AutoGPTEmbeddedChat
        apiBaseURL="http://localhost:8006"
        brandName="Forwarding Digital"
        getAccessToken={vi.fn().mockResolvedValue("embed-token")}
        title="Forwarding Assistant"
      />,
    );

    expect(
      screen.getByRole("heading", { name: "Forwarding Assistant" }),
    ).toBeDefined();
    expect(
      await screen.findByLabelText("Message Forwarding Assistant"),
    ).toBeDefined();
    expect(screen.queryByText("Daily operations")).toBeNull();
    fireEvent.click(screen.getByRole("button", { name: "Open chat sessions" }));
    expect(screen.getByText("Daily operations")).toBeDefined();
    expect(screen.getByText("1 automation block enabled")).toBeDefined();
    expect(screen.getByRole("button", { name: /artifacts/i })).toBeDefined();
    expect(screen.queryByText(/sign in to autogpt/i)).toBeNull();
  });

  it("uses host branding and prompt suggestions", async () => {
    render(
      <AutoGPTEmbeddedChat
        apiBaseURL="http://localhost:8006"
        brandName="Relay Freight OS"
        getAccessToken={vi.fn().mockResolvedValue("embed-token")}
        suggestedPrompts={["Create a Monday exception report"]}
        title="Operations Copilot"
      />,
    );

    const composer = await screen.findByLabelText("Message Operations Copilot");
    fireEvent.click(
      screen.getByRole("button", {
        name: "Create a Monday exception report",
      }),
    );
    expect((composer as HTMLTextAreaElement).value).toBe(
      "Create a Monday exception report",
    );
  });

  it("delegates assistant links to the host application", async () => {
    const onNavigate = vi.fn();
    sessionChatMock.messages = [
      {
        id: "assistant-link",
        role: "assistant",
        parts: [
          { type: "text", text: "[Open agent](/library/agents/agent-1)" },
        ],
      },
    ];

    render(
      <AutoGPTEmbeddedChat
        apiBaseURL="http://localhost:8006"
        getAccessToken={vi.fn().mockResolvedValue("embed-token")}
        onNavigate={onNavigate}
      />,
    );

    await screen.findByLabelText("Message Assistant");
    fireEvent.click(screen.getByRole("link", { name: "Open agent" }));
    expect(onNavigate).toHaveBeenCalledWith("/library/agents/agent-1");
  });

  it("groups adjacent assistant segments and docks artifacts in the header", async () => {
    sessionChatMock.messages = [
      {
        id: "user-1",
        role: "user",
        parts: [{ type: "text", text: "Yeah" }],
      },
      {
        id: "empty-assistant",
        role: "assistant",
        parts: [{ type: "text", text: " " }],
      },
      {
        id: "reasoning",
        role: "assistant",
        parts: [{ type: "reasoning", text: "Checking", state: "done" }],
      },
      {
        id: "answer",
        role: "assistant",
        parts: [
          {
            type: "text",
            text: "Done ![pixel](https://tracker.test/pixel.png)",
          },
        ],
      },
    ];

    render(
      <AutoGPTEmbeddedChat
        apiBaseURL="http://localhost:8006"
        brandName="Forwarding Digital"
        getAccessToken={vi.fn().mockResolvedValue("embed-token")}
        title="Forwarding Assistant"
      />,
    );

    await screen.findByLabelText("Message Forwarding Assistant");
    expect(screen.getAllByText("Forwarding Assistant")).toHaveLength(2);
    const artifactButton = screen.getByRole("button", {
      name: "Artifacts (1)",
    });
    expect(artifactButton.closest("header")).not.toBeNull();
    expect(document.querySelector("img")).toBeNull();
  });
});
