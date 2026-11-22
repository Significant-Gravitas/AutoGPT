import { server } from "@/mocks/mock-server";
import {
  cleanup,
  fireEvent,
  render,
  screen,
} from "@/tests/integrations/test-utils";
import { http, HttpResponse } from "msw";
import { afterEach, describe, expect, it, vi } from "vitest";
import output from "../../../../tools/DesignPage/__tests__/pageOutput.json";
import {
  APPLY_MESSAGE,
  DISCARD_MESSAGE,
} from "../../../../tools/DesignPage/helpers";
import { CopilotChatActionsProvider } from "../../../CopilotChatActionsProvider/CopilotChatActionsProvider";
import type { MessagePart } from "../../helpers";
import { ChainMessageParts } from "../ChainMessageParts";

vi.mock("../../../ToolChain/ToolChain", () => ({
  ToolChain: () => <div data-testid="tool-chain" />,
}));

function pagePart(overrides: Record<string, unknown> = {}) {
  return {
    type: "tool-design_page",
    toolCallId: "call-1",
    state: "output-available",
    input: {},
    output,
    ...overrides,
  } as unknown as MessagePart;
}

function renderParts(parts: MessagePart[], canActOnCards = true) {
  const onSend = vi.fn();
  render(
    <CopilotChatActionsProvider onSend={onSend}>
      <ChainMessageParts
        parts={parts}
        messageID="message-1"
        isCurrentlyStreaming={false}
        canActOnCards={canActOnCards}
      />
    </CopilotChatActionsProvider>,
  );
  return { onSend };
}

describe("page preview card", () => {
  afterEach(cleanup);

  it("draws the page the backend resolved, outside the tool chain", () => {
    renderParts([pagePart()]);

    expect(screen.getByText("Page preview")).toBeDefined();
    expect(screen.getByText("Posts")).toBeDefined();
    expect(screen.getByText("Pick a keyword to draft.")).toBeDefined();
    expect(screen.getByText("remote work")).toBeDefined();
    expect(screen.queryByTestId("tool-chain")).toBeNull();
    // A preview runs nothing, so its row buttons stay off the table.
    expect(screen.queryByRole("button", { name: /^Draft/ })).toBeNull();
  });

  it.each([
    ["Apply", APPLY_MESSAGE],
    ["Discard", DISCARD_MESSAGE],
  ])("sends %s as the user's reply, once", (label, message) => {
    const { onSend } = renderParts([pagePart()]);

    fireEvent.click(screen.getByRole("button", { name: label }));
    fireEvent.click(screen.getByRole("button", { name: label }));

    expect(onSend).toHaveBeenCalledTimes(1);
    expect(onSend).toHaveBeenCalledWith(message);
  });

  it("links a saved page into Pages and offers no buttons", () => {
    const page = { ...output.page, id: "page-1", revision: 1 };
    server.use(
      http.get("*/api/pages", () =>
        HttpResponse.json({ pages: [listed("page-1", 1)] }),
      ),
    );
    renderParts([pagePart({ output: { ...output, saved: true, page } })]);

    expect(screen.getByText("Page saved")).toBeDefined();
    expect(screen.queryByRole("button", { name: "Apply" })).toBeNull();
    const link = screen.getByRole("link", { name: "Open it in Pages" });
    expect(link.getAttribute("href")).toBe("/pages/page-1");
  });

  it("refuses to apply a preview of a page that has moved on", async () => {
    const page = { ...output.page, id: "page-1", revision: 2 };
    server.use(
      http.get("*/api/pages", () =>
        HttpResponse.json({ pages: [listed("page-1", 3)] }),
      ),
    );
    renderParts([pagePart({ output: { ...output, page } })]);

    expect(
      await screen.findByText(/A newer version of this page exists/),
    ).toBeDefined();
    expect(screen.queryByRole("button", { name: "Apply" })).toBeNull();
  });
});

function listed(id: string, revision: number) {
  return {
    id,
    title: "Posts",
    revision,
    workflow_count: 2,
    expert_count: 0,
    updated_at: "2026-11-21T12:00:00Z",
  };
}
