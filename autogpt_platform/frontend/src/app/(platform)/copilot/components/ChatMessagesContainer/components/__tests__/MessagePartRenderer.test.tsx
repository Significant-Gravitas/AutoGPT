import { afterEach, describe, expect, it, vi } from "vitest";
import { cleanup, render, screen } from "@testing-library/react";
import type { UIDataTypes, UIMessage, UITools } from "ai";
import { MessagePartRenderer } from "../MessagePartRenderer";

type Part = UIMessage<unknown, UIDataTypes, UITools>["parts"][number];

vi.mock("../ReasoningCollapse", () => ({
  ReasoningCollapse: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="reasoning-collapse">{children}</div>
  ),
}));

// The real MessageResponse renders markdown via Streamdown, passing the custom
// `components.img` override (WorkspaceMediaImage) down. We exercise that override
// directly so its video/image/empty-src branches are covered.
type ImgComponent = (
  props: React.JSX.IntrinsicElements["img"],
) => React.ReactNode;
vi.mock("@/components/ai-elements/message", () => ({
  MessageResponse: ({
    children,
    components,
  }: {
    children: React.ReactNode;
    components?: { img?: ImgComponent };
  }) => {
    const Img = components?.img;
    return (
      <div data-testid="message-response">
        {children}
        {Img ? (
          <div data-testid="custom-img-outputs">
            {Img({ src: "/v.mp4", alt: "video:Clip" })}
            {Img({ src: "/p.png", alt: "Pic" })}
            {Img({ src: "", alt: "missing" })}
          </div>
        ) : null}
      </div>
    );
  },
}));

vi.mock("@/components/molecules/ErrorCard/ErrorCard", () => ({
  ErrorCard: () => null,
}));

vi.mock("@/services/feature-flags/use-get-flag", () => ({
  Flag: { ARTIFACTS: "artifacts" },
  useGetFlag: () => false,
}));

vi.mock("@phosphor-icons/react", async (importOriginal) => {
  const actual = (await importOriginal()) as Record<string, unknown>;
  return { ...actual, ExclamationMarkIcon: () => null };
});

vi.mock("../StoppedTaskCard", () => ({
  StoppedTaskCard: () => <div data-testid="stopped-task-card" />,
}));

function stubTool(name: string, testid: string) {
  return { [name]: () => <div data-testid={testid} /> };
}
vi.mock("../../../../components/ArtifactCard/ArtifactCard", () => ({
  ArtifactCard: () => <div data-testid="artifact-card" />,
}));
vi.mock("../../../../tools/AskQuestion/AskQuestion", () =>
  stubTool("AskQuestionTool", "tool-ask-question"),
);
vi.mock("../../../../tools/ConnectIntegrationTool/ConnectIntegrationTool", () =>
  stubTool("ConnectIntegrationTool", "tool-connect-integration"),
);
vi.mock("../../../../tools/CreateAgent/CreateAgent", () =>
  stubTool("CreateAgentTool", "tool-create-agent"),
);
vi.mock("../../../../tools/DecomposeGoal/DecomposeGoal", () =>
  stubTool("DecomposeGoalTool", "tool-decompose-goal"),
);
vi.mock("../../../../tools/EditAgent/EditAgent", () =>
  stubTool("EditAgentTool", "tool-edit-agent"),
);
vi.mock("../../../../tools/FeatureRequests/FeatureRequests", () => ({
  CreateFeatureRequestTool: () => <div data-testid="tool-create-fr" />,
  SearchFeatureRequestsTool: () => <div data-testid="tool-search-fr" />,
}));
vi.mock("../../../../tools/FindAgents/FindAgents", () =>
  stubTool("FindAgentsTool", "tool-find-agents"),
);
vi.mock("../../../../tools/FolderTool/FolderTool", () =>
  stubTool("FolderTool", "tool-folder"),
);
vi.mock("../../../../tools/FindBlocks/FindBlocks", () =>
  stubTool("FindBlocksTool", "tool-find-blocks"),
);
vi.mock("../../../../tools/GenericTool/GenericTool", () =>
  stubTool("GenericTool", "tool-generic"),
);
vi.mock("../../../../tools/RunAgent/RunAgent", () =>
  stubTool("RunAgentTool", "tool-run-agent"),
);
vi.mock("../../../../tools/RunBlock/RunBlock", () =>
  stubTool("RunBlockTool", "tool-run-block"),
);
vi.mock("../../../../tools/RunMCPTool/RunMCPTool", () =>
  stubTool("RunMCPToolComponent", "tool-run-mcp"),
);
vi.mock("../../../../tools/SearchDocs/SearchDocs", () =>
  stubTool("SearchDocsTool", "tool-search-docs"),
);
vi.mock("../../../../tools/SetupTrigger/SetupTrigger", () =>
  stubTool("SetupTriggerTool", "tool-setup-trigger"),
);
vi.mock("../../../../tools/ViewAgentOutput/ViewAgentOutput", () =>
  stubTool("ViewAgentOutputTool", "tool-view-agent-output"),
);

function toolPart(type: string): Part {
  return {
    type,
    state: "output-available",
    toolCallId: `call-${type}`,
    output: {},
  } as unknown as Part;
}

describe("MessagePartRenderer reasoning branch", () => {
  afterEach(() => {
    cleanup();
  });

  it("renders a ReasoningCollapse wrapping a <pre> with the reasoning text", () => {
    const part = {
      type: "reasoning",
      text: "step-by-step plan",
      state: "done",
    } as unknown as Part;

    render(<MessagePartRenderer part={part} messageID="m1" partIndex={0} />);

    const collapse = screen.getByTestId("reasoning-collapse");
    expect(collapse).toBeDefined();
    const pre = collapse.querySelector("pre");
    expect(pre).not.toBeNull();
    expect(pre?.textContent).toBe("step-by-step plan");
  });

  it("returns null when the reasoning text is whitespace-only", () => {
    const part = {
      type: "reasoning",
      text: "   \n  ",
      state: "done",
    } as unknown as Part;

    const { container } = render(
      <MessagePartRenderer part={part} messageID="m1" partIndex={0} />,
    );
    expect(container.firstChild).toBeNull();
    expect(screen.queryByTestId("reasoning-collapse")).toBeNull();
  });

  it("returns null when the reasoning part has no text key", () => {
    const part = { type: "reasoning", state: "done" } as unknown as Part;

    const { container } = render(
      <MessagePartRenderer part={part} messageID="m1" partIndex={0} />,
    );
    expect(container.firstChild).toBeNull();
    expect(screen.queryByTestId("reasoning-collapse")).toBeNull();
  });

  it("returns null when the reasoning part's text is not a string", () => {
    const part = {
      type: "reasoning",
      text: 42,
      state: "done",
    } as unknown as Part;

    const { container } = render(
      <MessagePartRenderer part={part} messageID="m1" partIndex={0} />,
    );
    expect(container.firstChild).toBeNull();
    expect(screen.queryByTestId("reasoning-collapse")).toBeNull();
  });

  it("renders reasoning content when it contains non-whitespace surrounded by whitespace", () => {
    const part = {
      type: "reasoning",
      text: "  reasoning-with-pad  ",
      state: "done",
    } as unknown as Part;

    render(<MessagePartRenderer part={part} messageID="m2" partIndex={3} />);

    const pre = screen.getByTestId("reasoning-collapse").querySelector("pre");
    expect(pre?.textContent).toBe("  reasoning-with-pad  ");
  });
});

describe("MessagePartRenderer text branch", () => {
  afterEach(() => {
    cleanup();
  });

  it("renders plain text through the MessageResponse renderer", () => {
    const part = { type: "text", text: "hello world" } as unknown as Part;
    render(<MessagePartRenderer part={part} messageID="m1" partIndex={0} />);
    expect(screen.getByTestId("message-response").textContent).toContain(
      "hello world",
    );
  });

  it("renders a <video> for video: alt, an <img> for image alt, and nothing for empty src", () => {
    const part = { type: "text", text: "media" } as unknown as Part;
    const { container } = render(
      <MessagePartRenderer part={part} messageID="m1" partIndex={0} />,
    );
    const outputs = screen.getByTestId("custom-img-outputs");
    // video: alt -> <video><source/></video>
    expect(outputs.querySelector("video source")?.getAttribute("src")).toBe(
      "/v.mp4",
    );
    // normal alt -> <img>
    const img = outputs.querySelector("img");
    expect(img?.getAttribute("src")).toBe("/p.png");
    expect(img?.getAttribute("alt")).toBe("Pic");
    // empty src -> render nothing extra (only the one video + one img above)
    expect(container.querySelectorAll("img")).toHaveLength(1);
  });

  it("renders inline ArtifactCards when artifacts are forced on", () => {
    const fileId = "550e8400-e29b-41d4-a716-446655440000";
    const part = {
      type: "text",
      text: `Here is [report](workspace://${fileId}).`,
    } as unknown as Part;
    render(
      <MessagePartRenderer
        part={part}
        messageID="m1"
        partIndex={0}
        forceArtifacts
      />,
    );
    expect(screen.getByTestId("artifact-card")).toBeDefined();
  });

  it("renders the StoppedTaskCard for a cancellation marker", () => {
    const part = {
      type: "text",
      text: "[__COPILOT_ERROR_f7a1__] Operation cancelled",
    } as unknown as Part;
    render(<MessagePartRenderer part={part} messageID="m1" partIndex={0} />);
    expect(screen.getByTestId("stopped-task-card")).toBeDefined();
  });

  it("renders a system-marker note", () => {
    const part = {
      type: "text",
      text: "[__COPILOT_SYSTEM_e3b0__] Reconnected to the session",
    } as unknown as Part;
    render(<MessagePartRenderer part={part} messageID="m1" partIndex={0} />);
    expect(screen.getByText("Reconnected to the session")).toBeDefined();
  });
});

describe("MessagePartRenderer tool dispatch", () => {
  afterEach(() => {
    cleanup();
  });

  it.each([
    ["tool-ask_question", "tool-ask-question"],
    ["tool-find_block", "tool-find-blocks"],
    ["tool-find_agent", "tool-find-agents"],
    ["tool-find_library_agent", "tool-find-agents"],
    ["tool-search_docs", "tool-search-docs"],
    ["tool-get_doc_page", "tool-search-docs"],
    ["tool-connect_integration", "tool-connect-integration"],
    ["tool-run_block", "tool-run-block"],
    ["tool-continue_run_block", "tool-run-block"],
    ["tool-run_mcp_tool", "tool-run-mcp"],
    ["tool-run_agent", "tool-run-agent"],
    ["tool-schedule_agent", "tool-run-agent"],
    ["tool-setup_agent_webhook_trigger", "tool-setup-trigger"],
    ["tool-decompose_goal", "tool-decompose-goal"],
    ["tool-create_agent", "tool-create-agent"],
    ["tool-edit_agent", "tool-edit-agent"],
    ["tool-view_agent_output", "tool-view-agent-output"],
    ["tool-search_feature_requests", "tool-search-fr"],
    ["tool-create_feature_request", "tool-create-fr"],
    ["tool-create_folder", "tool-folder"],
    ["tool-list_folders", "tool-folder"],
    ["tool-update_folder", "tool-folder"],
    ["tool-move_folder", "tool-folder"],
    ["tool-delete_folder", "tool-folder"],
    ["tool-move_agents_to_folder", "tool-folder"],
  ])("dispatches %s to its renderer", (type, testid) => {
    render(
      <MessagePartRenderer
        part={toolPart(type)}
        messageID="m1"
        partIndex={0}
      />,
    );
    expect(screen.getByTestId(testid)).toBeDefined();
  });

  it("renders the GenericTool for an unrecognised tool- type", () => {
    render(
      <MessagePartRenderer
        part={toolPart("tool-some_builtin")}
        messageID="m1"
        partIndex={0}
      />,
    );
    expect(screen.getByTestId("tool-generic")).toBeDefined();
  });

  it("returns null for a non-tool, non-text/reasoning part type", () => {
    const part = { type: "step-start" } as unknown as Part;
    const { container } = render(
      <MessagePartRenderer part={part} messageID="m1" partIndex={0} />,
    );
    expect(container.firstChild).toBeNull();
  });
});
