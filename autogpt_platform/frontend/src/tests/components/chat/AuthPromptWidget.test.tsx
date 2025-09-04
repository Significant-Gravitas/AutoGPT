import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import { AuthPromptWidget } from "@/components/chat/AuthPromptWidget";

// Mock next/navigation
const mockPush = jest.fn();
const mockReplace = jest.fn();
jest.mock("next/navigation", () => ({
  useRouter: () => ({
    push: mockPush,
    replace: mockReplace,
    back: jest.fn(),
    forward: jest.fn(),
    refresh: jest.fn(),
    prefetch: jest.fn(),
  }),
  usePathname: () => "/marketplace/discover",
  useSearchParams: () => new URLSearchParams(),
}));

describe("AuthPromptWidget", () => {
  const defaultProps = {
    message: "Please sign in to continue",
    sessionId: "session_123",
  };

  beforeEach(() => {
    jest.clearAllMocks();
    // Clear localStorage
    localStorage.clear();
  });

  it("should render authentication prompt correctly", () => {
    render(<AuthPromptWidget {...defaultProps} />);

    expect(screen.getByText("Authentication Required")).toBeInTheDocument();
    expect(
      screen.getByText("Sign in to set up and manage agents"),
    ).toBeInTheDocument();
    expect(screen.getByText(defaultProps.message)).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /Sign In/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /Create Account/i }),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/Your chat session will be preserved/),
    ).toBeInTheDocument();
  });

  it("should display agent info when provided", () => {
    const propsWithAgent = {
      ...defaultProps,
      agentInfo: {
        graph_id: "graph_123",
        name: "Test Agent",
        trigger_type: "schedule" as const,
      },
    };

    render(<AuthPromptWidget {...propsWithAgent} />);

    expect(screen.getByText(/Ready to set up:/)).toBeInTheDocument();
    expect(screen.getByText("Test Agent")).toBeInTheDocument();
    expect(screen.getByText(/Type:/)).toBeInTheDocument();
    expect(screen.getByText("schedule")).toBeInTheDocument();
  });

  it("should store session info in localStorage when Sign In clicked", () => {
    const agentInfo = {
      graph_id: "graph_123",
      name: "Test Agent",
      trigger_type: "schedule" as const,
    };

    render(<AuthPromptWidget {...defaultProps} agentInfo={agentInfo} />);

    fireEvent.click(screen.getByRole("button", { name: /Sign In/i }));

    expect(localStorage.getItem("pending_chat_session")).toBe("session_123");
    expect(localStorage.getItem("pending_agent_setup")).toBe(
      JSON.stringify(agentInfo),
    );
  });

  it("should navigate to login page with return URL when Sign In clicked", () => {
    render(<AuthPromptWidget {...defaultProps} />);

    fireEvent.click(screen.getByRole("button", { name: /Sign In/i }));

    const expectedReturnUrl = encodeURIComponent(
      "/marketplace/discover?sessionId=session_123",
    );
    expect(mockPush).toHaveBeenCalledWith(
      `/login?returnUrl=${expectedReturnUrl}`,
    );
  });

  it("should navigate to signup page with return URL when Create Account clicked", () => {
    render(<AuthPromptWidget {...defaultProps} />);

    fireEvent.click(screen.getByRole("button", { name: /Create Account/i }));

    const expectedReturnUrl = encodeURIComponent(
      "/marketplace/discover?sessionId=session_123",
    );
    expect(mockPush).toHaveBeenCalledWith(
      `/signup?returnUrl=${expectedReturnUrl}`,
    );
  });

  it("should use custom return URL when provided", () => {
    render(<AuthPromptWidget {...defaultProps} returnUrl="/custom/path" />);

    fireEvent.click(screen.getByRole("button", { name: /Sign In/i }));

    const expectedReturnUrl = encodeURIComponent(
      "/custom/path?sessionId=session_123",
    );
    expect(mockPush).toHaveBeenCalledWith(
      `/login?returnUrl=${expectedReturnUrl}`,
    );
  });

  it("should apply custom className when provided", () => {
    const { container } = render(
      <AuthPromptWidget {...defaultProps} className="custom-class" />,
    );

    const widget = container.querySelector(".custom-class");
    expect(widget).toBeInTheDocument();
  });

  it("should not store agent info if not provided", () => {
    render(<AuthPromptWidget {...defaultProps} />);

    fireEvent.click(screen.getByRole("button", { name: /Sign In/i }));

    expect(localStorage.getItem("pending_chat_session")).toBe("session_123");
    expect(localStorage.getItem("pending_agent_setup")).toBeNull();
  });
});
