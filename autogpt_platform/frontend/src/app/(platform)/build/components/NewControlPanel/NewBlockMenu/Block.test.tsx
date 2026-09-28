import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { expect, test, vi } from "vitest";
import type { BlockInfo } from "@/app/api/__generated__/models/blockInfo";
import { TooltipProvider } from "@/components/ui/tooltip";
import { Block } from "./Block";

const addBlock = vi.fn();

vi.mock("@xyflow/react", () => ({
  useReactFlow: () => ({
    fitView: vi.fn(),
    getViewport: () => ({ x: 0, y: 0, zoom: 1 }),
  }),
  useStoreApi: () => ({
    getState: () => ({ width: 1024, height: 768 }),
  }),
}));

vi.mock("../../../stores/controlPanelStore", () => ({
  useControlPanelStore: (selector: (state: object) => unknown) =>
    selector({ setBlockMenuOpen: vi.fn() }),
}));

vi.mock("../../../stores/nodeStore", () => ({
  useNodeStore: (selector?: (state: object) => unknown) => {
    const state = { addBlock, updateNodeData: vi.fn() };
    return selector ? selector(state) : state;
  },
}));

function disabledBlock(): BlockInfo {
  return {
    id: "disabled-block",
    name: "DisabledBlock",
    description: "Unavailable block",
    inputSchema: {},
    outputSchema: {},
    costs: [],
    categories: [],
    contributors: [],
    staticOutput: false,
    uiType: "standard",
    disabled: true,
    disabledReason: "Missing local configuration",
  } as BlockInfo;
}

test("disables unavailable blocks and explains why on hover", async () => {
  const user = userEvent.setup();
  render(
    <TooltipProvider>
      <Block blockData={disabledBlock()} title="DisabledBlock" />
    </TooltipProvider>,
  );

  const button = screen.getByRole("button", { name: /disabled/i });
  expect(button).toHaveProperty("disabled", true);

  await user.hover(button.parentElement!);

  expect((await screen.findByRole("tooltip")).textContent).toContain(
    "Missing local configuration",
  );
  expect(addBlock).not.toHaveBeenCalled();
});
