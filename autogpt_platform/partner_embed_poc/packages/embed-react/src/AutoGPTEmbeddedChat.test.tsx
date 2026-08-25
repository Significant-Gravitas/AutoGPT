import type { UIMessage } from "ai";
import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { AutoGPTEmbeddedChat } from "./AutoGPTEmbeddedChat";
import { getEmbedSession, updateEmbedSessionTitle } from "./session-api";

const sessionChatMock = vi.hoisted(() => ({
  messages: [] as UIMessage[],
  error: null as Error | null,
  sendMessage: vi.fn(),
}));

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
  updateEmbedSessionTitle: vi
    .fn()
    .mockResolvedValue("Compare the active shipment lanes and..."),
}));

vi.mock("./useSessionChat", () => ({
  useSessionChat: () => ({
    messages: sessionChatMock.messages,
    sendMessage: sessionChatMock.sendMessage,
    stop: vi.fn(),
    status: "ready",
    error: sessionChatMock.error,
  }),
}));

describe("AutoGPTEmbeddedChat", () => {
  beforeEach(() => {
    cleanup();
    vi.clearAllMocks();
    sessionChatMock.messages = [];
    sessionChatMock.error = null;
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
  it("describes only the capabilities granted to a restricted session", async () => {
    vi.mocked(getEmbedSession).mockResolvedValueOnce({
      id: "session-1",
      title: null,
      createdAt: "2026-08-24T10:00:00Z",
      updatedAt: "2026-08-24T11:00:00Z",
      chatStatus: "idle",
      messages: [],
      hasMoreMessages: false,
      oldestSequence: null,
      capabilities: ["jobs.read", "documents.read"],
    });

    render(
      <AutoGPTEmbeddedChat
        apiBaseURL="http://localhost:8006"
        getAccessToken={vi.fn().mockResolvedValue("embed-token")}
        title="Operations Copilot"
      />,
    );

    expect(
      await screen.findByText(
        "You can review arrivals and investigate exceptions and read session documents.",
      ),
    ).toBeDefined();
    expect(screen.queryByText(/run enabled blocks/i)).toBeNull();
    expect(screen.queryByText(/create documents/i)).toBeNull();
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

  it("relabels a raw internal agent path without changing its destination", async () => {
    const onNavigate = vi.fn();
    sessionChatMock.messages = [
      {
        id: "assistant-raw-link",
        role: "assistant",
        parts: [
          {
            type: "text",
            text: "[/library/agents/agent-1](/library/agents/agent-1)",
          },
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
    fireEvent.click(screen.getByRole("link", { name: "Open saved agent" }));
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

  it("titles a quota-rejected session from its first user message", async () => {
    vi.mocked(getEmbedSession).mockResolvedValueOnce({
      id: "session-1",
      title: null,
      createdAt: "2026-08-24T10:00:00Z",
      updatedAt: "2026-08-24T11:00:00Z",
      chatStatus: "idle",
      messages: [],
      hasMoreMessages: false,
      oldestSequence: null,
      capabilities: ["documents.read"],
    });
    sessionChatMock.messages = [
      {
        id: "failed-user-message",
        role: "user",
        parts: [
          {
            type: "text",
            text: "Compare the active shipment lanes and flag the highest risk",
          },
        ],
      },
    ];
    sessionChatMock.error = new Error("usage limit");

    render(
      <AutoGPTEmbeddedChat
        apiBaseURL="http://localhost:8006"
        getAccessToken={vi.fn().mockResolvedValue("embed-token")}
        title="Operations Copilot"
      />,
    );

    await screen.findByLabelText("Message Operations Copilot");
    await waitFor(() =>
      expect(updateEmbedSessionTitle).toHaveBeenCalledWith(
        "http://localhost:8006",
        "session-1",
        "Compare the active shipment lanes and flag the highest risk",
        expect.any(Function),
      ),
    );
    fireEvent.click(screen.getByRole("button", { name: "Open chat sessions" }));
    expect(
      screen.getByText("Compare the active shipment lanes and..."),
    ).toBeDefined();
  });
});
