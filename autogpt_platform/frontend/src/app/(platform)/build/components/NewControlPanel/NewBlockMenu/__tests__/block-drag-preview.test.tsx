import { ReactFlowProvider } from "@xyflow/react";
import { describe, expect, test, vi } from "vitest";
import {
  createEvent,
  fireEvent,
  render,
  screen,
} from "@/tests/integrations/test-utils";

import type { BlockInfo } from "@/app/api/__generated__/models/blockInfo";
import { useControlPanelStore } from "../../../../stores/controlPanelStore";
import { Block } from "../Block";
import { IntegrationBlock } from "../IntergrationBlock";

const blockData: BlockInfo = {
  id: "b1ab9b19-67a6-406d-abf5-2dba76d00c79",
  name: "CalculatorBlock",
  description: "Performs a mathematical operation on two numbers.",
  inputSchema: {},
  outputSchema: {},
  costs: [],
  categories: [],
  contributors: [],
  staticOutput: false,
  uiType: "Standard",
};

function dragStart(card: HTMLElement) {
  const event = createEvent.dragStart(card, {
    dataTransfer: {
      effectAllowed: "",
      setData: vi.fn(),
      setDragImage: vi.fn(),
    },
  });
  fireEvent(card, event);
  return (event as DragEvent).dataTransfer as unknown as {
    effectAllowed: string;
    setData: ReturnType<typeof vi.fn>;
    setDragImage: ReturnType<typeof vi.fn>;
  };
}

describe("block drag preview", () => {
  test.each([
    ["Block", () => <Block title="Calculator Block" blockData={blockData} />],
    [
      "IntegrationBlock",
      () => <IntegrationBlock title="Calculator Block" blockData={blockData} />,
    ],
  ])(
    "%s drags its block with a role-classed preview named after it",
    (_, ui) => {
      useControlPanelStore.setState({ blockMenuOpen: true });
      render(<ReactFlowProvider>{ui()}</ReactFlowProvider>);

      const card = screen.getByText("Calculator").closest("[draggable]")!;
      const dataTransfer = dragStart(card as HTMLElement);

      expect(dataTransfer.effectAllowed).toBe("copy");
      expect(dataTransfer.setData).toHaveBeenCalledWith(
        "application/reactflow",
        JSON.stringify(blockData),
      );
      const preview = dataTransfer.setDragImage.mock.calls[0][0] as HTMLElement;
      expect(preview.textContent).toBe("Calculator");
      expect(preview.className).toBe("text-body font-medium leading-inherit");
      expect(preview.style.fontSize).toBe("");
      expect(preview.style.position).toBe("absolute");
      expect(useControlPanelStore.getState().blockMenuOpen).toBe(false);
    },
  );
});
