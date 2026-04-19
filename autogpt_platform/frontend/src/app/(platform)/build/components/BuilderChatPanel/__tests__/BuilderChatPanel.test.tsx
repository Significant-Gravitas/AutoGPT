import {
  render,
  screen,
  fireEvent,
  cleanup,
} from "@/tests/integrations/test-utils";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { BuilderChatPanel } from "../BuilderChatPanel";

vi.mock("../useBuilderChatPanel", () => ({
  useBuilderChatPanel: vi.fn(),
}));

import { useBuilderChatPanel } from "../useBuilderChatPanel";

const mockUseBuilderChatPanel = vi.mocked(useBuilderChatPanel);

function makeMockHook(
  overrides: Partial<ReturnType<typeof useBuilderChatPanel>> = {},
): ReturnType<typeof useBuilderChatPanel> {
  return {
    isOpen: false,
    handleToggle: vi.fn(),
    panelRef: undefined,
    sessionId: null,
    flowID: null,
    flowVersion: null,
    messages: [],
    status: "ready",
    error: undefined,
    stop: vi.fn(),
    onSend: vi.fn(),
    queuedMessages: [],
    isCreatingSession: false,
    isBootstrapping: false,
    revertTargetVersion: null,
    handleRevert: vi.fn(),
    ...overrides,
  } as ReturnType<typeof useBuilderChatPanel>;
}

beforeEach(() => {
  mockUseBuilderChatPanel.mockReturnValue(makeMockHook());
});

afterEach(() => {
  cleanup();
});

describe("BuilderChatPanel", () => {
  it("renders the toggle button when closed", () => {
    render(<BuilderChatPanel />);
    expect(screen.getByLabelText("Chat with builder")).toBeDefined();
  });

  it("does not render the panel content when closed", () => {
    render(<BuilderChatPanel />);
    expect(screen.queryByText("Chat with Builder")).toBeNull();
  });

  it("calls handleToggle when the toggle button is clicked", () => {
    const handleToggle = vi.fn();
    mockUseBuilderChatPanel.mockReturnValue(makeMockHook({ handleToggle }));
    render(<BuilderChatPanel />);
    fireEvent.click(screen.getByLabelText("Chat with builder"));
    expect(handleToggle).toHaveBeenCalledOnce();
  });

  it("renders the panel header when open", () => {
    mockUseBuilderChatPanel.mockReturnValue(makeMockHook({ isOpen: true }));
    render(<BuilderChatPanel />);
    expect(screen.getByText("Chat with Builder")).toBeDefined();
    expect(screen.getByRole("complementary")).toBeDefined();
  });

  it("shows bootstrapping state when isBootstrapping is true", () => {
    mockUseBuilderChatPanel.mockReturnValue(
      makeMockHook({ isOpen: true, isBootstrapping: true }),
    );
    render(<BuilderChatPanel />);
    expect(screen.getByText(/Preparing builder chat/i)).toBeDefined();
  });

  it("shows the Revert button when a revert target is available", () => {
    const handleRevert = vi.fn();
    mockUseBuilderChatPanel.mockReturnValue(
      makeMockHook({ isOpen: true, revertTargetVersion: 3, handleRevert }),
    );
    render(<BuilderChatPanel />);
    const revert = screen.getByRole("button", { name: /Revert to version 3/i });
    expect(revert).toBeDefined();
    fireEvent.click(revert);
    expect(handleRevert).toHaveBeenCalledOnce();
  });

  it("does not show the Revert button when revertTargetVersion is null", () => {
    mockUseBuilderChatPanel.mockReturnValue(
      makeMockHook({ isOpen: true, revertTargetVersion: null }),
    );
    render(<BuilderChatPanel />);
    expect(screen.queryByRole("button", { name: /Revert/i })).toBeNull();
  });
});
