import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { AutoGPTEmbeddedChat } from "./AutoGPTEmbeddedChat";

vi.mock("./useEmbeddedChat", () => ({
  useEmbeddedChat: () => ({
    messages: [],
    sendMessage: vi.fn(),
    stop: vi.fn(),
    status: "ready",
    error: null,
    initializationError: null,
    isInitialized: true,
  }),
}));

describe("AutoGPTEmbeddedChat", () => {
  it("renders a branded chat shell without asking for AutoGPT credentials", () => {
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
    expect(screen.getByLabelText("Message Forwarding Assistant")).toBeDefined();
    expect(screen.getByText("Forwarding Digital")).toBeDefined();
    expect(screen.queryByText(/sign in to autogpt/i)).toBeNull();
  });
});
