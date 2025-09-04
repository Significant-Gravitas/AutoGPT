import React from "react";
import { render, screen, fireEvent } from "@testing-library/react";
import { AgentSetupCard } from "@/components/chat/AgentSetupCard";

// Mock window.open
const mockOpen = jest.fn();
window.open = mockOpen;

// Mock navigator.clipboard
Object.assign(navigator, {
  clipboard: {
    writeText: jest.fn().mockResolvedValue(undefined),
  },
});

describe("AgentSetupCard", () => {
  const baseScheduleProps = {
    status: "success",
    triggerType: "schedule" as const,
    name: "Daily Email Processor",
    graphId: "graph_123",
    graphVersion: 1,
    scheduleId: "schedule_456",
    cron: "0 9 * * *",
    cronUtc: "0 14 * * *",
    timezone: "America/New_York",
    nextRun: new Date(Date.now() + 86400000).toISOString(),
    addedToLibrary: true,
    libraryId: "lib_789",
    message: "Successfully scheduled agent to run daily",
  };

  const baseWebhookProps = {
    status: "success",
    triggerType: "webhook" as const,
    name: "Webhook Agent",
    graphId: "graph_456",
    graphVersion: 2,
    webhookUrl: "https://api.autogpt.com/webhooks/abc123",
    addedToLibrary: true,
    libraryId: "lib_101",
    message: "Successfully created webhook trigger",
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe("Schedule Setup", () => {
    it("should render schedule setup card correctly", () => {
      render(<AgentSetupCard {...baseScheduleProps} />);

      expect(screen.getByText("Agent Setup Complete")).toBeInTheDocument();
      expect(screen.getByText("Daily Email Processor")).toBeInTheDocument();
      expect(screen.getByText(baseScheduleProps.message)).toBeInTheDocument();
      expect(screen.getByText("Scheduled Execution")).toBeInTheDocument();
    });

    it("should display schedule details", () => {
      render(<AgentSetupCard {...baseScheduleProps} />);

      expect(screen.getByText("Schedule:")).toBeInTheDocument();
      expect(screen.getByText("0 9 * * *")).toBeInTheDocument();
      expect(screen.getByText("Timezone:")).toBeInTheDocument();
      expect(screen.getByText("America/New_York")).toBeInTheDocument();
      expect(screen.getByText("Next run:")).toBeInTheDocument();
    });

    it("should show library status when added", () => {
      render(<AgentSetupCard {...baseScheduleProps} />);

      expect(screen.getByText("Added to your library")).toBeInTheDocument();
    });

    it("should render View in Library button", () => {
      render(<AgentSetupCard {...baseScheduleProps} />);

      const libraryButton = screen.getByRole("button", {
        name: /View in Library/i,
      });
      expect(libraryButton).toBeInTheDocument();
    });

    it("should render View Runs button for schedules", () => {
      render(<AgentSetupCard {...baseScheduleProps} />);

      const runsButton = screen.getByRole("button", { name: /View Runs/i });
      expect(runsButton).toBeInTheDocument();
    });

    it("should open library page when View in Library clicked", () => {
      render(<AgentSetupCard {...baseScheduleProps} />);

      fireEvent.click(screen.getByRole("button", { name: /View in Library/i }));

      expect(mockOpen).toHaveBeenCalledWith(
        "/library/agents/lib_789",
        "_blank",
      );
    });

    it("should open runs page with schedule ID when View Runs clicked", () => {
      render(<AgentSetupCard {...baseScheduleProps} />);

      fireEvent.click(screen.getByRole("button", { name: /View Runs/i }));

      expect(mockOpen).toHaveBeenCalledWith(
        "/library/runs?scheduleId=schedule_456",
        "_blank",
      );
    });
  });

  describe("Webhook Setup", () => {
    it("should render webhook setup card correctly", () => {
      render(<AgentSetupCard {...baseWebhookProps} />);

      expect(screen.getByText("Agent Setup Complete")).toBeInTheDocument();
      expect(screen.getByText("Webhook Agent")).toBeInTheDocument();
      expect(screen.getByText(baseWebhookProps.message)).toBeInTheDocument();
      expect(screen.getByText("Webhook Trigger")).toBeInTheDocument();
    });

    it("should display webhook URL with copy button", () => {
      render(<AgentSetupCard {...baseWebhookProps} />);

      expect(screen.getByText("Webhook URL:")).toBeInTheDocument();
      expect(screen.getByText(baseWebhookProps.webhookUrl)).toBeInTheDocument();
      expect(screen.getByRole("button", { name: /Copy/i })).toBeInTheDocument();
    });

    it("should copy webhook URL to clipboard when Copy clicked", async () => {
      render(<AgentSetupCard {...baseWebhookProps} />);

      fireEvent.click(screen.getByRole("button", { name: /Copy/i }));

      expect(navigator.clipboard.writeText).toHaveBeenCalledWith(
        baseWebhookProps.webhookUrl,
      );
    });

    it("should not show View Runs button for webhooks", () => {
      render(<AgentSetupCard {...baseWebhookProps} />);

      expect(
        screen.queryByRole("button", { name: /View Runs/i }),
      ).not.toBeInTheDocument();
    });
  });

  describe("Failed Setup", () => {
    it("should render failed setup card correctly", () => {
      const failedProps = {
        ...baseScheduleProps,
        status: "error",
        message: "Failed to set up agent: Invalid cron expression",
      };

      render(<AgentSetupCard {...failedProps} />);

      expect(screen.getByText("Setup Failed")).toBeInTheDocument();
      expect(screen.getByText(failedProps.message)).toBeInTheDocument();
    });

    it("should not show action buttons on failure", () => {
      const failedProps = {
        ...baseScheduleProps,
        status: "error",
      };

      render(<AgentSetupCard {...failedProps} />);

      expect(
        screen.queryByRole("button", { name: /View in Library/i }),
      ).not.toBeInTheDocument();
      expect(
        screen.queryByRole("button", { name: /View Runs/i }),
      ).not.toBeInTheDocument();
    });
  });

  describe("Additional Info", () => {
    it("should display agent ID and version", () => {
      render(<AgentSetupCard {...baseScheduleProps} />);

      expect(screen.getByText(/Agent ID:/)).toBeInTheDocument();
      expect(screen.getByText("graph_123")).toBeInTheDocument();
      expect(screen.getByText(/Version: 1/)).toBeInTheDocument();
    });

    it("should display schedule ID when present", () => {
      render(<AgentSetupCard {...baseScheduleProps} />);

      expect(screen.getByText(/Schedule ID:/)).toBeInTheDocument();
      expect(screen.getByText("schedule_456")).toBeInTheDocument();
    });

    it("should format next run time correctly", () => {
      const nextRun = new Date("2024-12-25T14:00:00Z").toISOString();
      const props = {
        ...baseScheduleProps,
        nextRun,
      };

      render(<AgentSetupCard {...props} />);

      // Check that next run is displayed (exact format depends on locale)
      expect(screen.getByText(/Next run:/)).toBeInTheDocument();
    });
  });

  describe("Edge Cases", () => {
    it("should handle missing optional props", () => {
      const minimalProps = {
        status: "success",
        triggerType: "schedule" as const,
        name: "Minimal Agent",
        graphId: "graph_min",
        graphVersion: 1,
        message: "Setup complete",
      };

      render(<AgentSetupCard {...minimalProps} />);

      expect(screen.getByText("Agent Setup Complete")).toBeInTheDocument();
      expect(screen.getByText("Minimal Agent")).toBeInTheDocument();
    });

    it("should open default library page when no libraryId", () => {
      const propsNoLibraryId = {
        ...baseScheduleProps,
        libraryId: undefined,
      };

      render(<AgentSetupCard {...propsNoLibraryId} />);

      fireEvent.click(screen.getByRole("button", { name: /View in Library/i }));

      expect(mockOpen).toHaveBeenCalledWith("/library", "_blank");
    });

    it("should open default runs page when no scheduleId", () => {
      const propsNoScheduleId = {
        ...baseScheduleProps,
        scheduleId: undefined,
      };

      render(<AgentSetupCard {...propsNoScheduleId} />);

      fireEvent.click(screen.getByRole("button", { name: /View Runs/i }));

      expect(mockOpen).toHaveBeenCalledWith("/library/runs", "_blank");
    });

    it("should apply custom className when provided", () => {
      const { container } = render(
        <AgentSetupCard {...baseScheduleProps} className="custom-class" />,
      );

      const card = container.querySelector(".custom-class");
      expect(card).toBeInTheDocument();
    });
  });
});
