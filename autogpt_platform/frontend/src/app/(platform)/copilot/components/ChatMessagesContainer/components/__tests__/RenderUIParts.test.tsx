import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import output from "../../../../tools/RenderUI/__tests__/renderUIOutput.json";
import { CopilotChatActionsProvider } from "../../../CopilotChatActionsProvider/CopilotChatActionsProvider";
import type { MessagePart } from "../../helpers";
import { ChainMessageParts } from "../ChainMessageParts";

vi.mock("../../../ToolChain/ToolChain", () => ({
  ToolChain: () => <div data-testid="tool-chain" />,
}));

vi.mock("@/services/analytics/posthog-capture", () => ({
  capturePostHogEvent: vi.fn(),
}));

function toolPart(overrides: Record<string, unknown>) {
  return {
    type: "tool-render_ui",
    toolCallId: "call-1",
    state: "output-available",
    input: {},
    output,
    ...overrides,
  } as unknown as MessagePart;
}

interface RenderArgs {
  parts: MessagePart[];
  canActOnCards?: boolean;
  isCurrentlyStreaming?: boolean;
  readOnly?: boolean;
  fileUrlBuilder?: (fileId: string) => string;
}

function renderParts({
  parts,
  canActOnCards = true,
  isCurrentlyStreaming = false,
  readOnly,
  fileUrlBuilder,
}: RenderArgs) {
  const onSend = vi.fn();
  render(
    <CopilotChatActionsProvider onSend={onSend}>
      <ChainMessageParts
        parts={parts}
        messageID="message-1"
        isCurrentlyStreaming={isCurrentlyStreaming}
        canActOnCards={canActOnCards}
        readOnly={readOnly}
        fileUrlBuilder={fileUrlBuilder}
      />
    </CopilotChatActionsProvider>,
  );
  return { onSend };
}

function withBlocks(blocks: unknown[]) {
  return toolPart({ output: { type: "ui_blocks", message: "", blocks } });
}

describe("render_ui cards", () => {
  afterEach(cleanup);

  it("draws what the backend tool returned, outside the tool chain", () => {
    renderParts({ parts: [toolPart({})] });

    expect(screen.getByText("Leads")).toBeDefined();
    expect(screen.getAllByText("412")).toHaveLength(2);
    expect(screen.getByText("+37%")).toBeDefined();
    expect(screen.getByText("Where leads drop out")).toBeDefined();
    expect(screen.getByText("31 failed")).toBeDefined();
    expect(screen.getByText("New subject lines")).toBeDefined();
    expect(
      screen.getByRole("img", {
        name: "Replies per week. This month: 14, 17, 28, 37",
      }),
    ).toBeDefined();
    expect(screen.queryByTestId("tool-chain")).toBeNull();
  });

  it("sends an option's title as the reply", () => {
    const { onSend } = renderParts({ parts: [toolPart({})] });

    fireEvent.click(screen.getByRole("button", { name: /^Fix enrichment/ }));

    expect(onSend).toHaveBeenCalledWith("Fix enrichment");
  });

  it("sends an option once when clicked twice", () => {
    const { onSend } = renderParts({ parts: [toolPart({})] });
    const button = screen.getByRole("button", { name: /^Fix enrichment/ });

    fireEvent.click(button);
    fireEvent.click(button);

    expect(onSend).toHaveBeenCalledTimes(1);
  });

  it.each([
    ["a shared chat", { readOnly: true }],
    ["an older message", { canActOnCards: false }],
    ["a turn still running", { isCurrentlyStreaming: true }],
  ])("disables options in %s", (_name, args) => {
    const { onSend } = renderParts({ parts: [toolPart({})], ...args });
    const button = screen.getByRole<HTMLButtonElement>("button", {
      name: /^Fix enrichment/,
    });

    fireEvent.click(button);

    expect(button.disabled).toBe(true);
    expect(onSend).not.toHaveBeenCalled();
  });

  it("skips unknown blocks and renders the rest", () => {
    renderParts({
      parts: [
        withBlocks([
          { type: "hologram", frames: [] },
          { type: "chart" },
          {
            type: "stats",
            items: [{ label: "Runs", value: "7" }],
          },
        ]),
      ],
    });

    expect(screen.getByText("Runs")).toBeDefined();
  });

  it("renders nothing for output it cannot read", () => {
    const { container } = render(
      <CopilotChatActionsProvider onSend={vi.fn()}>
        <ChainMessageParts
          parts={[toolPart({ output: "{not json" })]}
          messageID="message-1"
          isCurrentlyStreaming={false}
        />
      </CopilotChatActionsProvider>,
    );

    expect(container.textContent).toBe("");
  });

  it("shows a skeleton while the call is running", () => {
    renderParts({
      parts: [toolPart({ state: "input-available", output: undefined })],
      isCurrentlyStreaming: true,
    });

    expect(screen.getByTestId("render-ui-skeleton")).toBeDefined();
  });

  it("shows nothing once the stream has ended without output", () => {
    renderParts({
      parts: [toolPart({ state: "input-available", output: undefined })],
    });

    expect(screen.queryByTestId("render-ui-skeleton")).toBeNull();
  });

  it("builds gallery images through the given file URL builder", () => {
    renderParts({
      parts: [toolPart({})],
      fileUrlBuilder: (fileID) => `/share/files/${fileID}`,
    });

    expect(screen.getByAltText("Before").getAttribute("src")).toBe(
      "/share/files/file-1",
    );
  });

  it("treats a script URL in block text as plain words", () => {
    renderParts({
      parts: [
        withBlocks([
          {
            type: "stats",
            items: [{ label: "javascript:alert(1)", value: "1" }],
          },
        ]),
      ],
    });

    expect(screen.getByText("javascript:alert(1)")).toBeDefined();
    expect(screen.queryByRole("link")).toBeNull();
  });

  it("keeps a wide table scrolling inside its card", () => {
    renderParts({
      parts: [
        withBlocks([
          { type: "table", columns: ["Metric"], rows: [["Meetings"]] },
        ]),
      ],
    });

    const scroller = screen.getByTestId("render-ui-table-scroll");
    expect(scroller.className).toContain("overflow-x-auto");
    expect(scroller.textContent).toContain("Meetings");
  });
});
