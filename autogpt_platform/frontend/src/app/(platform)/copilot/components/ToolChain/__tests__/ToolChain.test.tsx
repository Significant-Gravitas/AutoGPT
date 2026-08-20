import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { act, cleanup, render, screen } from "@/tests/integrations/test-utils";
import { useCopilotUIStore } from "@/app/(platform)/copilot/store";
import { CopilotChatActionsProvider } from "../../CopilotChatActionsProvider/CopilotChatActionsProvider";
import type { MessagePart } from "../../ChatMessagesContainer/helpers";
import type { PendingQuestions } from "../../QuestionDock/helpers";
import { PendingQuestionsContext } from "../../QuestionDock/PendingQuestionsContext";
import { ToolChain } from "../ToolChain";

vi.mock("../../SetupRequirementsCard/SetupRequirementsCard", async () => {
  const { useContext, useEffect } = await import("react");
  const { ChainActionsContext } = await import("../chainActions");

  interface MockProps {
    output: { setup_info?: { agent_name?: string } };
  }

  function SetupRequirementsCard({ output }: MockProps) {
    const chainActions = useContext(ChainActionsContext);
    const name = output.setup_info?.agent_name ?? "Integration";
    useEffect(() => {
      if (!chainActions) return;
      chainActions.register({
        id: name,
        ready: name !== "NotReady",
        buildMessage: () =>
          name === "Silent" ? null : `Connected ${name}. Please continue.`,
      });
      return () => chainActions.unregister(name);
    }, [chainActions, name]);
    return <div>{`setup-card-${name}`}</div>;
  }

  return { SetupRequirementsCard };
});

function reasoningPart(text: string): MessagePart {
  return { type: "reasoning", text, state: "done" } as MessagePart;
}

function toolPart(
  toolName: string,
  state: string,
  extras: Record<string, unknown> = {},
): MessagePart {
  return {
    type: `tool-${toolName}`,
    state,
    toolCallId: `call-${toolName}`,
    input: {},
    output: undefined,
    ...extras,
  } as MessagePart;
}

function getPanel(header: HTMLElement): HTMLElement | null {
  const panelId = header.getAttribute("aria-controls");
  return panelId ? document.getElementById(panelId) : null;
}

describe("ToolChain", () => {
  afterEach(() => {
    cleanup();
    useCopilotUIStore.setState({ initialPrompt: null, sentMessageCount: 0 });
  });

  it("renders nothing when no parts map to chain rows", () => {
    const { container } = render(<ToolChain parts={[]} isStreaming={false} />);

    expect(container.textContent).toBe("");
  });

  it("collapses a settled chain to a summary heading and expands on click", async () => {
    const user = userEvent.setup();
    render(
      <ToolChain
        parts={[
          reasoningPart("Compare the options first"),
          toolPart("web_search", "output-available", {
            input: { query: "copilot" },
            output: { results: [] },
          }),
          toolPart("bash_exec", "output-available", {
            input: { command: "ls" },
            output: "files",
          }),
        ]}
        isStreaming={false}
      />,
    );

    const header = screen.getByRole("button", {
      name: /thought it through, searched the web, ran commands/i,
    });
    expect(header.getAttribute("aria-expanded")).toBe("false");
    expect(getPanel(header)?.getAttribute("aria-hidden")).toBe("true");

    await user.click(header);

    expect(header.getAttribute("aria-expanded")).toBe("true");
    expect(getPanel(header)?.getAttribute("aria-hidden")).toBe("false");
    expect(screen.getByText("Done")).toBeDefined();
    expect(screen.getByText('Searched the web for "copilot"')).toBeDefined();

    await user.click(header);
    expect(header.getAttribute("aria-expanded")).toBe("false");
  });

  it("expands a reasoning row to reveal the thought stream", async () => {
    const user = userEvent.setup();
    render(
      <ToolChain
        parts={[reasoningPart("Compare the options first")]}
        isStreaming={false}
      />,
    );

    await user.click(
      screen.getByRole("button", { name: /thought it through/i }),
    );

    const reasoningRow = screen.getByRole("button", { name: "Thought" });
    expect(reasoningRow.getAttribute("aria-expanded")).toBe("false");

    await user.click(reasoningRow);

    expect(reasoningRow.getAttribute("aria-expanded")).toBe("true");
    expect(screen.getByText("Compare the options first")).toBeDefined();
  });

  it("shows only the sliding window of latest rows while streaming", () => {
    render(
      <ToolChain
        parts={[
          reasoningPart("Compare the options first"),
          toolPart("web_search", "output-available", {
            input: { query: "copilot" },
            output: { results: [] },
          }),
          toolPart("bash_exec", "input-available", {
            input: { command: "ls" },
          }),
        ]}
        isStreaming
      />,
    );

    expect(screen.getAllByText('Running command "ls"…').length).toBeGreaterThan(
      0,
    );
    expect(screen.queryByText("Thought")).toBeNull();
    expect(screen.queryByText("Done")).toBeNull();

    const header = screen.getByRole("button", {
      name: /running command "ls"/i,
    });
    expect(header.getAttribute("aria-expanded")).toBe("true");
  });

  it("streams live reasoning inline without needing a click", () => {
    render(
      <ToolChain
        parts={[
          {
            type: "reasoning",
            text: "Weighing the trade-offs",
            state: "streaming",
          } as MessagePart,
        ]}
        isStreaming
      />,
    );

    expect(screen.getAllByText("Thinking…").length).toBeGreaterThan(0);
    expect(screen.getByText("Weighing the trade-offs")).toBeDefined();
  });

  it("renders the provider icon when the tool output names a provider", async () => {
    const user = userEvent.setup();
    const { container } = render(
      <ToolChain
        parts={[
          toolPart("run_block", "output-available", {
            input: { block_name: "Send Email" },
            output: { provider: "github", ok: true },
          }),
        ]}
        isStreaming={false}
      />,
    );

    await user.click(screen.getByRole("button", { name: /ran blocks/i }));

    const icon = container.querySelector('img[src*="/integrations/github"]');
    expect(icon).not.toBeNull();
  });

  it("auto-expands browser rows while the artifact panel is open", async () => {
    const user = userEvent.setup();
    const { artifactPanel } = useCopilotUIStore.getState();
    useCopilotUIStore.setState({
      artifactPanel: { ...artifactPanel, isOpen: true },
    });

    render(
      <ToolChain
        parts={[
          toolPart("browser_navigate", "output-available", {
            input: { url: "https://agpt.co" },
            output: { title: "AutoGPT" },
          }),
        ]}
        isStreaming={false}
      />,
    );

    await user.click(screen.getByRole("button", { name: /browsed the web/i }));

    const row = screen.getByRole("button", {
      name: /opened "https:\/\/agpt.co"/i,
    });
    expect(row.getAttribute("aria-expanded")).toBe("true");

    useCopilotUIStore.setState({
      artifactPanel: { ...artifactPanel, isOpen: false },
    });
  });

  it("surfaces the latest error in the heading and drops the Done step", async () => {
    const user = userEvent.setup();
    render(
      <ToolChain
        parts={[
          toolPart("web_search", "output-error", {
            input: { query: "copilot" },
            errorText: "Rate limited",
          }),
        ]}
        isStreaming={false}
      />,
    );

    expect(screen.getAllByText("Rate limited").length).toBeGreaterThan(0);

    await user.click(screen.getByRole("button", { name: /rate limited/i }));

    expect(screen.queryByText("Done")).toBeNull();
    expect(
      screen.getByText('Failed while searching the web for "copilot"'),
    ).toBeDefined();
  });

  it("keeps action-required rows visible while the chain is collapsed", () => {
    render(
      <ToolChain
        parts={[
          toolPart("web_search", "output-available", {
            input: { query: "copilot" },
            output: { results: [] },
          }),
          toolPart("run_block", "output-available", {
            output: { type: "review_required", block_name: "Send Email" },
          }),
        ]}
        isStreaming={false}
      />,
    );

    const header = screen.getByRole("button", {
      name: /review send email/i,
    });
    expect(getPanel(header)?.getAttribute("aria-hidden")).toBe("false");

    expect(screen.getAllByText("Review Send Email").length).toBeGreaterThan(1);
    expect(screen.queryByText('Searched the web for "copilot"')).toBeNull();
    expect(screen.getByText("Send Email")).toBeDefined();
  });

  it("drafts answered questions into the chat input and dismisses on send", async () => {
    const user = userEvent.setup();
    const pending: PendingQuestions = {
      dockId: "m1:call-ask_question",
      callIds: ["call-ask_question"],
      questions: [
        { question: "Which region?", keyword: "region", example: "Europe" },
      ],
    };

    render(
      <CopilotChatActionsProvider onSend={vi.fn()}>
        <PendingQuestionsContext.Provider value={pending}>
          <ToolChain
            parts={[
              toolPart("ask_question", "output-available", {
                output: {
                  type: "agent_builder_clarification_needed",
                  questions: pending.questions,
                },
              }),
            ]}
            isStreaming={false}
          />
        </PendingQuestionsContext.Provider>
      </CopilotChatActionsProvider>,
    );

    expect(screen.getByText("Answer a few questions")).toBeDefined();
    expect(screen.getByText("Which region?")).toBeDefined();

    const submit = screen.getByRole("button", {
      name: "Add answers to message",
    });
    expect(submit.hasAttribute("disabled")).toBe(true);

    await user.type(
      screen.getByPlaceholderText("e.g. Europe"),
      "Western Europe",
    );
    expect(submit.hasAttribute("disabled")).toBe(false);

    await user.click(submit);

    expect(useCopilotUIStore.getState().initialPrompt).toBe(
      "**Here are my answers:**\n\n> Which region?\n\nWestern Europe\n\nPlease proceed.",
    );

    act(() => useCopilotUIStore.getState().notifyMessageSent());

    expect(screen.queryByText("Answer a few questions")).toBeNull();
  });

  it("offers a Proceed step for confirm-only setup cards", async () => {
    const user = userEvent.setup();
    render(
      <ToolChain
        parts={[
          toolPart("connect_integration", "output-available", {
            output: {
              type: "setup_requirements",
              setup_info: { agent_name: "GitHub" },
            },
          }),
        ]}
        isStreaming={false}
      />,
    );

    expect(screen.getByText("setup-card-GitHub")).toBeDefined();
    expect(
      screen.getByText("Everything's filled in — send it to continue"),
    ).toBeDefined();

    const proceed = screen.getByRole("button", { name: "Proceed" });
    expect(proceed.hasAttribute("disabled")).toBe(false);

    await user.click(proceed);

    expect(useCopilotUIStore.getState().initialPrompt).toBe(
      "Connected GitHub. Please continue.",
    );
  });

  it("disables Proceed until every registered card is ready", () => {
    render(
      <ToolChain
        parts={[
          toolPart("connect_integration", "output-available", {
            output: {
              type: "setup_requirements",
              setup_info: { agent_name: "NotReady" },
            },
          }),
        ]}
        isStreaming={false}
      />,
    );

    expect(
      screen.getByText("Complete the steps above, then send to continue"),
    ).toBeDefined();
    expect(
      screen.getByRole("button", { name: "Proceed" }).hasAttribute("disabled"),
    ).toBe(true);
    expect(useCopilotUIStore.getState().initialPrompt).toBeNull();
  });

  it("does not draft anything when ready cards produce no message", async () => {
    const user = userEvent.setup();
    render(
      <ToolChain
        parts={[
          toolPart("connect_integration", "output-available", {
            output: {
              type: "setup_requirements",
              setup_info: { agent_name: "Silent" },
            },
          }),
        ]}
        isStreaming={false}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Proceed" }));

    expect(useCopilotUIStore.getState().initialPrompt).toBeNull();
  });
});
