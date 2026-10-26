import { describe, expect, it } from "vitest";
import type { UIDataTypes, UIMessage, UITools } from "ai";
import { render, screen } from "@/tests/integrations/test-utils";
import { MessagePartRenderer } from "../../../components/ChatMessagesContainer/components/MessagePartRenderer";
import { isChainableToolPart } from "../../../components/ChatMessagesContainer/helpers";

type Part = UIMessage<unknown, UIDataTypes, UITools>["parts"][number];

function viaRunCapability(output: unknown, state = "output-available"): Part {
  return {
    type: "tool-run_capability",
    toolCallId: "call-1",
    state,
    input: {
      id: "tool:request_integration",
      input: { service: "Gelato", use_case: "Print sticker sheets" },
    },
    output,
  } as Part;
}

const RECORDED = {
  type: "integration_request",
  message: "Gelato is not supported yet.",
  service: "Gelato",
  recorded: true,
};

function renderPart(part: Part) {
  return render(
    <MessagePartRenderer part={part} messageID="m-1" partIndex={0} />,
  );
}

describe("IntegrationRequestCard", () => {
  it("says the service isn't supported and that the request was kept", () => {
    renderPart(viaRunCapability(RECORDED));

    expect(screen.getByText("Gelato isn't supported yet")).toBeDefined();
    expect(
      screen.getByText(/added your request to the list of connections/),
    ).toBeDefined();
  });

  it("says what happens next when the request could not be recorded", () => {
    renderPart(viaRunCapability({ ...RECORDED, recorded: false }));

    expect(
      screen.getByText(/Otto can still help with the rest of the task/),
    ).toBeDefined();
    expect(screen.queryByText(/added your request/)).toBeNull();
  });

  it("renders the direct tool call the same way, from a stored JSON string", () => {
    renderPart({
      type: "tool-request_integration",
      toolCallId: "call-2",
      state: "output-available",
      input: { service: "Gelato" },
      output: JSON.stringify(RECORDED),
    } as Part);

    expect(screen.getByText("Gelato isn't supported yet")).toBeDefined();
  });

  it("shows nothing until the answer has landed", () => {
    const { container } = renderPart(
      viaRunCapability(undefined, "input-available"),
    );

    expect(container.textContent).toBe("");
  });

  it("still shows a settled answer that is not the card, via capability", () => {
    const { container } = renderPart(
      viaRunCapability({
        type: "error",
        message: "Printify can be connected: find_capability(query='printify')",
      }),
    );

    expect(container.textContent).not.toBe("");
    expect(screen.queryByText(/isn't supported yet/)).toBeNull();
  });

  it("still shows a failed direct call", () => {
    const { container } = renderPart({
      type: "tool-request_integration",
      toolCallId: "call-3",
      state: "output-error",
      input: { service: "Gelato" },
      errorText: "Tool failed",
    } as Part);

    expect(container.textContent).not.toBe("");
  });

  it("stays out of the collapsing tool chain", () => {
    expect(isChainableToolPart(viaRunCapability(RECORDED))).toBe(false);
    expect(
      isChainableToolPart({
        ...viaRunCapability(RECORDED),
        input: { id: "tool:create_feature_request", input: {} },
      } as Part),
    ).toBe(true);
  });
});
