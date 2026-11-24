import {
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { Toaster } from "@/components/molecules/Toast/toaster";
import { ApiError } from "@/lib/autogpt-server-api/helpers";
import { render as renderWithProviders } from "@/tests/integrations/test-utils";
import { afterEach, describe, expect, it, vi } from "vitest";
import { CopilotChatActionsContext } from "../../CopilotChatActionsProvider/useCopilotChatActions";
import {
  FixResultCard,
  PlanSteps,
  QuestionsCard,
  SetupCard,
  SkillCard,
  SuggestedGoalCard,
  TriggerSetupCard,
  ValidationCard,
} from "../InfoCards";

describe("PlanSteps", () => {
  afterEach(cleanup);

  it("renders step descriptions across statuses with block chips", () => {
    render(
      <PlanSteps
        steps={[
          {
            description: "Fetch data",
            status: "completed",
            block_name: "HTTP",
          },
          { description: "Summarize", status: "in_progress" },
          { description: "Send email", status: "pending" },
        ]}
      />,
    );

    expect(screen.getByText("Fetch data")).toBeDefined();
    expect(screen.getByText("Summarize")).toBeDefined();
    expect(screen.getByText("Send email")).toBeDefined();
    expect(screen.getByText("HTTP")).toBeDefined();
  });

  it("falls back to inline JSON for steps without a description", () => {
    render(<PlanSteps steps={[{ step_id: "s1" }]} />);

    expect(screen.getByText('{"step_id":"s1"}')).toBeDefined();
  });
});

describe("ValidationCard", () => {
  afterEach(cleanup);

  it("shows a valid status for valid graphs", () => {
    render(<ValidationCard output={{ valid: true }} />);

    expect(screen.getByText("Workflow is valid")).toBeDefined();
  });

  it("lists validation errors when present", () => {
    render(
      <ValidationCard
        output={{ valid: false, errors: ["Missing input", "Broken link"] }}
      />,
    );

    expect(screen.getByText("Missing input")).toBeDefined();
    expect(screen.getByText("Broken link")).toBeDefined();
  });

  it("shows a generic failure when errors are absent", () => {
    render(<ValidationCard output={{ valid: false }} />);

    expect(screen.getByText("Workflow has errors")).toBeDefined();
  });
});

describe("FixResultCard", () => {
  afterEach(cleanup);

  it("summarizes applied fixes when the graph is valid after fixing", () => {
    render(
      <FixResultCard
        output={{
          valid_after_fix: true,
          fixes_applied: ["Linked input", "Removed node"],
        }}
      />,
    );

    expect(screen.getByText("Fixed — applied 2 fixes")).toBeDefined();
  });

  it("lists remaining errors when the graph is still broken", () => {
    render(
      <FixResultCard
        output={{ valid_after_fix: false, remaining_errors: ["Still broken"] }}
      />,
    );

    expect(screen.getByText("1 error remaining")).toBeDefined();
    expect(screen.getByText("Still broken")).toBeDefined();
  });
});

describe("QuestionsCard", () => {
  afterEach(cleanup);

  it("renders questions with examples", () => {
    render(
      <QuestionsCard
        questions={[
          { question: "Which region?", example: "us-east" },
          { question: "What budget?" },
        ]}
      />,
    );

    expect(screen.getByText("Which region?")).toBeDefined();
    expect(screen.getByText("e.g. us-east")).toBeDefined();
    expect(screen.getByText("What budget?")).toBeDefined();
  });

  it("falls back to inline JSON for malformed entries", () => {
    render(<QuestionsCard questions={[{ keyword: "region" }]} />);

    expect(screen.getByText('{"keyword":"region"}')).toBeDefined();
  });
});

describe("SetupCard", () => {
  afterEach(cleanup);

  it("shows the integration name and connection prompt", () => {
    render(
      <SetupCard
        output={{ setup_info: { agent_name: "Notion" } }}
        provider={null}
      />,
    );

    expect(screen.getByText("Notion")).toBeDefined();
    expect(screen.getByText("Connection required")).toBeDefined();
  });

  it("renders nothing without an agent name", () => {
    const { container } = render(
      <SetupCard output={{ setup_info: {} }} provider={null} />,
    );

    expect(container.firstChild).toBeNull();
  });
});

describe("SkillCard", () => {
  afterEach(cleanup);

  it("shows the skill name, description and first trigger", () => {
    render(
      <SkillCard
        output={{
          name: "Weekly digest",
          description: "Summarizes the week",
          triggers: ["every friday", "on demand"],
        }}
      />,
    );

    expect(screen.getByText("Weekly digest")).toBeDefined();
    expect(screen.getByText("Summarizes the week")).toBeDefined();
    expect(screen.getByText("every friday")).toBeDefined();
    expect(screen.queryByText("on demand")).toBeNull();
  });

  it("renders nothing without a skill name", () => {
    const { container } = render(
      <SkillCard output={{ description: "orphan" }} />,
    );

    expect(container.firstChild).toBeNull();
  });
});

describe("SuggestedGoalCard", () => {
  afterEach(cleanup);

  it("renders the goal and reason without a send action", () => {
    render(
      <SuggestedGoalCard
        output={{ suggested_goal: "Build a scraper", reason: "You asked" }}
      />,
    );

    expect(screen.getByText("Build a scraper")).toBeDefined();
    expect(screen.getByText("You asked")).toBeDefined();
    expect(screen.queryByRole("button", { name: "Use this goal" })).toBeNull();
  });

  it("sends the goal through chat actions when clicked", () => {
    const onSend = vi.fn();
    render(
      <CopilotChatActionsContext.Provider
        value={{ onSend, chatSurface: "copilot" }}
      >
        <SuggestedGoalCard output={{ suggested_goal: "Build a scraper" }} />
      </CopilotChatActionsContext.Provider>,
    );

    fireEvent.click(screen.getByRole("button", { name: "Use this goal" }));

    expect(onSend).toHaveBeenCalledWith(
      "Please create a workflow with this goal: Build a scraper",
    );
  });

  it("shows a send failure and lets the user retry the same goal", async () => {
    const onSend = vi
      .fn()
      .mockRejectedValueOnce(new Error("Network error"))
      .mockResolvedValue(undefined);
    render(
      <CopilotChatActionsContext.Provider
        value={{ onSend, chatSurface: "copilot" }}
      >
        <SuggestedGoalCard
          output={{ suggested_goal: "Summarize weekly sales" }}
        />
        <Toaster />
      </CopilotChatActionsContext.Provider>,
    );

    const button = screen.getByRole("button", { name: "Use this goal" });
    fireEvent.click(button);

    expect(await screen.findByText("Couldn't send message")).toBeDefined();
    expect(
      screen.getByText(/check the chat before trying again/i),
    ).toBeDefined();
    expect(onSend).toHaveBeenCalledTimes(1);

    fireEvent.click(button);

    const message =
      "Please create a workflow with this goal: Summarize weekly sales";
    expect(onSend.mock.calls).toEqual([[message], [message]]);
  });

  it("ignores clicks while a send is pending and allows a retry after it fails", async () => {
    let rejectSend: (error: Error) => void = () => {};
    const onSend = vi
      .fn()
      .mockImplementationOnce(
        () =>
          new Promise<void>((_, reject) => {
            rejectSend = reject;
          }),
      )
      .mockResolvedValue(undefined);
    render(
      <CopilotChatActionsContext.Provider
        value={{ onSend, chatSurface: "copilot" }}
      >
        <SuggestedGoalCard
          output={{ suggested_goal: "Summarize weekly sales" }}
        />
        <Toaster />
      </CopilotChatActionsContext.Provider>,
    );

    const button = screen.getByRole("button", { name: "Use this goal" });
    fireEvent.click(button);
    fireEvent.click(button);
    expect(onSend).toHaveBeenCalledTimes(1);

    rejectSend(new Error("Network error"));
    expect(await screen.findByText("Couldn't send message")).toBeDefined();

    fireEvent.click(button);
    expect(onSend).toHaveBeenCalledTimes(2);
  });

  it("shows paywall recovery copy when sending requires an active plan", async () => {
    const onSend = vi
      .fn()
      .mockRejectedValueOnce(new ApiError("Payment required", 402, {}));
    // The paywall toast reads whether this install bills through React Query,
    // so it needs the app's providers.
    renderWithProviders(
      <CopilotChatActionsContext.Provider
        value={{ onSend, chatSurface: "copilot" }}
      >
        <SuggestedGoalCard
          output={{ suggested_goal: "Summarize weekly sales" }}
        />
        <Toaster />
      </CopilotChatActionsContext.Provider>,
    );

    fireEvent.click(screen.getByRole("button", { name: "Use this goal" }));

    expect(
      await screen.findByText("Sending needs an active plan"),
    ).toBeDefined();
    expect(
      screen.getByText(/the suggested goal is still on the card/i),
    ).toBeDefined();
  });

  it("renders nothing without a goal", () => {
    const { container } = render(<SuggestedGoalCard output={{}} />);

    expect(container.firstChild).toBeNull();
  });
});

describe("TriggerSetupCard", () => {
  afterEach(cleanup);

  it("shows the webhook URL and copies it", () => {
    const writeText = vi
      .spyOn(navigator.clipboard, "writeText")
      .mockResolvedValue(undefined);

    render(
      <TriggerSetupCard
        output={{
          message: "Webhook ready",
          webhook_url: "https://hooks.example.com/h1",
        }}
      />,
    );

    expect(screen.getByText("Webhook ready")).toBeDefined();
    expect(screen.getByText("https://hooks.example.com/h1")).toBeDefined();

    fireEvent.click(screen.getByRole("button", { name: "Copy" }));

    expect(writeText).toHaveBeenCalledWith("https://hooks.example.com/h1");
  });

  it("swallows clipboard failures on copy", async () => {
    const writeText = vi
      .spyOn(navigator.clipboard, "writeText")
      .mockRejectedValue(new Error("denied"));

    render(
      <TriggerSetupCard
        output={{ webhook_url: "https://hooks.example.com/h2" }}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "Copy" }));

    await waitFor(() =>
      expect(writeText).toHaveBeenCalledWith("https://hooks.example.com/h2"),
    );
  });

  it("falls back to a default message without a URL", () => {
    render(<TriggerSetupCard output={{}} />);

    expect(screen.getByText("Trigger is ready")).toBeDefined();
    expect(screen.queryByRole("button", { name: "Copy" })).toBeNull();
  });
});
