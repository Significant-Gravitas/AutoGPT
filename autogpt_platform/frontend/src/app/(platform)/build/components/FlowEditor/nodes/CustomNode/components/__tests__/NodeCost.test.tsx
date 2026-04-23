import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen } from "@/tests/integrations/test-utils";
import { NodeCost } from "../NodeCost";
import { BlockCost } from "@/app/api/__generated__/models/blockCost";
import { BlockCostType } from "@/app/api/__generated__/models/blockCostType";
import { useNodeStore } from "@/app/(platform)/build/stores/nodeStore";

vi.mock("@/hooks/useCredits", () => ({
  default: () => ({
    formatCredits: (credit: number | null) =>
      credit === null ? "-" : `$${(Math.abs(credit) / 100).toFixed(2)}`,
  }),
}));

function cost(overrides: Partial<BlockCost>): BlockCost {
  return {
    cost_amount: 100,
    cost_filter: {},
    cost_type: BlockCostType.run,
    ...overrides,
  } as BlockCost;
}

describe("NodeCost", () => {
  beforeEach(() => {
    useNodeStore.setState({ getHardCodedValues: () => ({}) } as any);
  });

  it("renders fixed /run label for RUN cost type", () => {
    render(<NodeCost blockCosts={[cost({})]} nodeId="n1" />);
    expect(screen.getByText("$1.00")).toBeTruthy();
    expect(screen.getByText(/\/run/)).toBeTruthy();
  });

  it("renders /sec for SECOND cost with divisor<=1", () => {
    render(
      <NodeCost
        blockCosts={[
          cost({
            cost_type: BlockCostType.second,
            cost_divisor: 1,
            cost_amount: 50,
          }),
        ]}
        nodeId="n1"
      />,
    );
    expect(screen.getByText(/\/sec/)).toBeTruthy();
  });

  it("renders '/ Ns' for SECOND cost with divisor>1", () => {
    render(
      <NodeCost
        blockCosts={[
          cost({
            cost_type: BlockCostType.second,
            cost_divisor: 10,
            cost_amount: 25,
          }),
        ]}
        nodeId="n1"
      />,
    );
    expect(screen.getByText(/\/ 10s/)).toBeTruthy();
  });

  it("renders /item for ITEMS with divisor<=1", () => {
    render(
      <NodeCost
        blockCosts={[cost({ cost_type: BlockCostType.items, cost_amount: 10 })]}
        nodeId="n1"
      />,
    );
    expect(screen.getByText(/\/item/)).toBeTruthy();
  });

  it("renders '/ N items' for ITEMS with divisor>1", () => {
    render(
      <NodeCost
        blockCosts={[
          cost({
            cost_type: BlockCostType.items,
            cost_divisor: 5,
            cost_amount: 40,
          }),
        ]}
        nodeId="n1"
      />,
    );
    expect(screen.getByText(/\/ 5 items/)).toBeTruthy();
  });

  it("renders '· by USD' for COST_USD and shows approximate prefix for positive floor", () => {
    render(
      <NodeCost
        blockCosts={[
          cost({ cost_type: BlockCostType.cost_usd, cost_amount: 200 }),
        ]}
        nodeId="n1"
      />,
    );
    expect(screen.getByText(/~\$2\.00/)).toBeTruthy();
    expect(screen.getByText(/· by USD/)).toBeTruthy();
  });

  it("renders em-dash when dynamic cost has zero floor", () => {
    render(
      <NodeCost
        blockCosts={[
          cost({ cost_type: BlockCostType.cost_usd, cost_amount: 0 }),
        ]}
        nodeId="n1"
      />,
    );
    expect(screen.getByText("—")).toBeTruthy();
  });

  it("renders '· by tokens' with floor hint for TOKENS", () => {
    render(
      <NodeCost
        blockCosts={[
          cost({ cost_type: BlockCostType.tokens, cost_amount: 150 }),
        ]}
        nodeId="n1"
      />,
    );
    expect(screen.getByText(/~\$1\.50/)).toBeTruthy();
    expect(screen.getByText(/· by tokens/)).toBeTruthy();
  });

  it("renders /byte suffix for BYTE cost type", () => {
    render(
      <NodeCost
        blockCosts={[cost({ cost_type: BlockCostType.byte, cost_amount: 1 })]}
        nodeId="n1"
      />,
    );
    expect(screen.getByText(/\/byte/)).toBeTruthy();
  });

  it("falls back to /<raw> suffix for unknown cost type", () => {
    render(
      <NodeCost
        blockCosts={[
          cost({
            // Simulate a future enum value not handled by the switch.
            cost_type: "experimental" as unknown as BlockCostType,
            cost_amount: 99,
          }),
        ]}
        nodeId="n1"
      />,
    );
    expect(screen.getByText(/\/experimental/)).toBeTruthy();
  });

  it("returns null when no matching blockCost found", () => {
    useNodeStore.setState({
      getHardCodedValues: () => ({ provider: "openai" }),
    } as any);
    const { container } = render(
      <NodeCost
        blockCosts={[
          cost({
            cost_filter: { provider: "anthropic" },
          }),
        ]}
        nodeId="n1"
      />,
    );
    expect(container.textContent).toBe("");
  });
});
