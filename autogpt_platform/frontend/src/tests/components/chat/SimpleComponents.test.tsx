import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import { AuthPromptWidget } from "@/components/chat/AuthPromptWidget";
import { CredentialsSetupWidget } from "@/components/chat/CredentialsSetupWidget";
import { AgentSetupCard } from "@/components/chat/AgentSetupCard";

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

// Mock window.open
window.open = jest.fn();

// Mock clipboard
Object.assign(navigator, {
  clipboard: {
    writeText: jest.fn().mockResolvedValue(undefined),
  },
});

describe("Chat Components", () => {
  describe("AuthPromptWidget", () => {
    it("renders authentication prompt", () => {
      render(
        <AuthPromptWidget
          message="Please sign in to continue"
          sessionId="test-session"
        />,
      );

      expect(screen.getByText("Authentication Required")).toBeInTheDocument();
      expect(
        screen.getByText("Please sign in to continue"),
      ).toBeInTheDocument();
      expect(
        screen.getByRole("button", { name: /Sign In/i }),
      ).toBeInTheDocument();
      expect(
        screen.getByRole("button", { name: /Create Account/i }),
      ).toBeInTheDocument();
    });

    it("displays agent info when provided", () => {
      render(
        <AuthPromptWidget
          message="Sign in required"
          sessionId="test-session"
          agentInfo={{
            graph_id: "graph_123",
            name: "Test Agent",
            trigger_type: "schedule",
          }}
        />,
      );

      expect(screen.getByText("Test Agent")).toBeInTheDocument();
      expect(screen.getByText("schedule")).toBeInTheDocument();
    });
  });

  describe("CredentialsSetupWidget", () => {
    it("renders credentials setup widget", () => {
      render(
        <CredentialsSetupWidget
          _agentId="agent_123"
          configuredCredentials={["github"]}
          missingCredentials={["openai", "slack"]}
          totalRequired={3}
        />,
      );

      expect(screen.getByText("Credentials Required")).toBeInTheDocument();
      expect(screen.getByText("1 of 3 configured")).toBeInTheDocument();
      expect(screen.getByText("GitHub")).toBeInTheDocument();
      expect(screen.getByText("OpenAI")).toBeInTheDocument();
      expect(screen.getByText("Slack")).toBeInTheDocument();
    });

    it("shows connect buttons for missing credentials", () => {
      render(
        <CredentialsSetupWidget
          _agentId="agent_123"
          configuredCredentials={[]}
          missingCredentials={["openai", "slack"]}
          totalRequired={2}
        />,
      );

      const connectButtons = screen.getAllByRole("button", {
        name: /Connect/i,
      });
      expect(connectButtons).toHaveLength(2);
    });
  });

  describe("AgentSetupCard", () => {
    it("renders successful schedule setup", () => {
      render(
        <AgentSetupCard
          status="success"
          triggerType="schedule"
          name="Daily Task"
          graphId="graph_123"
          graphVersion={1}
          cron="0 9 * * *"
          timezone="America/New_York"
          message="Successfully scheduled"
        />,
      );

      expect(screen.getByText("Agent Setup Complete")).toBeInTheDocument();
      expect(screen.getByText("Daily Task")).toBeInTheDocument();
      expect(screen.getByText("Successfully scheduled")).toBeInTheDocument();
      expect(screen.getByText("Scheduled Execution")).toBeInTheDocument();
      expect(screen.getByText("0 9 * * *")).toBeInTheDocument();
    });

    it("renders webhook setup", () => {
      render(
        <AgentSetupCard
          status="success"
          triggerType="webhook"
          name="Webhook Agent"
          graphId="graph_456"
          graphVersion={1}
          webhookUrl="https://api.example.com/webhook"
          message="Webhook created"
        />,
      );

      expect(screen.getByText("Webhook Trigger")).toBeInTheDocument();
      expect(
        screen.getByText("https://api.example.com/webhook"),
      ).toBeInTheDocument();
      expect(screen.getByRole("button", { name: /Copy/i })).toBeInTheDocument();
    });

    it("handles copy webhook URL", () => {
      render(
        <AgentSetupCard
          status="success"
          triggerType="webhook"
          name="Webhook Agent"
          graphId="graph_456"
          graphVersion={1}
          webhookUrl="https://api.example.com/webhook"
          message="Webhook created"
        />,
      );

      fireEvent.click(screen.getByRole("button", { name: /Copy/i }));
      expect(navigator.clipboard.writeText).toHaveBeenCalledWith(
        "https://api.example.com/webhook",
      );
    });
  });
});
