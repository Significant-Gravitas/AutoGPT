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

vi.mock("@/components/ai-elements/message", () => ({
  MessageResponse: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="message-response">{children}</div>
  ),
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

const stub = (name: string) => ({ [name]: () => null });
vi.mock("../../ArtifactCard/ArtifactCard", () => stub("ArtifactCard"));
vi.mock("../../../tools/AskQuestion/AskQuestion", () =>
  stub("AskQuestionTool"),
);
vi.mock("../../../tools/ConnectIntegrationTool/ConnectIntegrationTool", () =>
  stub("ConnectIntegrationTool"),
);
vi.mock("../../../tools/CreateAgent/CreateAgent", () =>
  stub("CreateAgentTool"),
);
vi.mock("../../../tools/EditAgent/EditAgent", () => stub("EditAgentTool"));
vi.mock("../../../tools/FeatureRequests/FeatureRequests", () => ({
  CreateFeatureRequestTool: () => null,
  SearchFeatureRequestsTool: () => null,
}));
vi.mock("../../../tools/FindAgents/FindAgents", () => stub("FindAgentsTool"));
vi.mock("../../../tools/FolderTool/FolderTool", () => stub("FolderTool"));
vi.mock("../../../tools/FindBlocks/FindBlocks", () => stub("FindBlocksTool"));
vi.mock("../../../tools/GenericTool/GenericTool", () => stub("GenericTool"));
vi.mock("../../../tools/RunAgent/RunAgent", () => stub("RunAgentTool"));
vi.mock("../../../tools/RunBlock/RunBlock", () => stub("RunBlockTool"));
vi.mock("../../../tools/RunMCPTool/RunMCPTool", () =>
  stub("RunMCPToolComponent"),
);
vi.mock("../../../tools/SearchDocs/SearchDocs", () => stub("SearchDocsTool"));
vi.mock("../../../tools/ViewAgentOutput/ViewAgentOutput", () =>
  stub("ViewAgentOutputTool"),
);

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
