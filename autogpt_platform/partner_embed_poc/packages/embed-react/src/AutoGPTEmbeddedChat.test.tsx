import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { AutoGPTEmbeddedChat } from "./AutoGPTEmbeddedChat";

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
    messages: [],
    sendMessage: vi.fn(),
    stop: vi.fn(),
    status: "ready",
    error: null,
  }),
}));

describe("AutoGPTEmbeddedChat", () => {
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
    expect(await screen.findByText("Daily operations")).toBeDefined();
    expect(
      await screen.findByLabelText("Message Forwarding Assistant"),
    ).toBeDefined();
    expect(screen.getByText("1 AutoGPT block enabled")).toBeDefined();
    expect(screen.getByRole("button", { name: /artifacts/i })).toBeDefined();
    expect(screen.queryByText(/sign in to autogpt/i)).toBeNull();
  });
});
