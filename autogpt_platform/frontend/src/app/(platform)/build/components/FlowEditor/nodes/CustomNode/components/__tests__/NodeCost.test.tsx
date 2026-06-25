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
    useNodeStore.setState({
      getHardCodedValues: () => ({}),
    } as Partial<ReturnType<typeof useNodeStore.getState>>);
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

  it("renders 'Pay-as-you-go' for COST_USD without a published rate", () => {
    render(
      <NodeCost
        blockCosts={[
          cost({ cost_type: BlockCostType.cost_usd, cost_amount: 200 }),
        ]}
        nodeId="n1"
      />,
    );
    expect(screen.getByText("Pay-as-you-go")).toBeTruthy();
    expect(screen.queryByText(/\$/)).toBeNull();
  });

  it("renders explicit in/out USD pair for COST_USD when published rate is provided", () => {
    render(
      <NodeCost
        blockCosts={[
          cost({
            cost_type: BlockCostType.cost_usd,
            cost_amount: 150,
            token_rate: {
              input_usd_per_1m: 0.3,
              output_usd_per_1m: 1.2,
            },
          }),
        ]}
        nodeId="n1"
      />,
    );
    expect(screen.getByText("$0.30 in / $1.20 out")).toBeTruthy();
    expect(screen.getByText(/per 1M tokens/)).toBeTruthy();
  });

  it("includes cache rates in the tooltip when present", () => {
    render(
      <NodeCost
        blockCosts={[
          cost({
            cost_type: BlockCostType.tokens,
            cost_amount: 14,
            token_rate: {
              input_usd_per_1m: 5,
              output_usd_per_1m: 25,
              cache_read_usd_per_1m: 0.5,
              cache_creation_usd_per_1m: 6.25,
            },
          }),
        ]}
        nodeId="n1"
      />,
    );
    const tooltipHost = screen
      .getByText("$5.00 in / $25.00 out")
      .closest("div");
    expect(tooltipHost?.getAttribute("title")).toMatch(/Cached input: \$0\.50/);
    expect(tooltipHost?.getAttribute("title")).toMatch(/Cache write: \$6\.25/);
  });

  it("omits cache rate lines from the tooltip when not provided", () => {
    render(
      <NodeCost
        blockCosts={[
          cost({
            cost_type: BlockCostType.tokens,
            cost_amount: 14,
            token_rate: {
              input_usd_per_1m: 5,
              output_usd_per_1m: 25,
              cache_read_usd_per_1m: null,
              cache_creation_usd_per_1m: null,
            },
          }),
        ]}
        nodeId="n1"
      />,
    );
    const tooltipHost = screen
      .getByText("$5.00 in / $25.00 out")
      .closest("div");
    const title = tooltipHost?.getAttribute("title") ?? "";
    expect(title).not.toMatch(/Cached input/);
    expect(title).not.toMatch(/Cache write/);
    expect(title).toMatch(/Input: \$5\.00/);
    expect(title).toMatch(/Output: \$25\.00/);
  });

  it("renders explicit in/out USD pair for TOKENS with published rates", () => {
    render(
      <NodeCost
        blockCosts={[
          cost({
            cost_type: BlockCostType.tokens,
            cost_amount: 14,
            token_rate: {
              input_usd_per_1m: 5,
              output_usd_per_1m: 25,
            },
          }),
        ]}
        nodeId="n1"
      />,
    );
    expect(screen.getByText("$5.00 in / $25.00 out")).toBeTruthy();
    expect(screen.getByText(/per 1M tokens/)).toBeTruthy();
  });

  it("falls back to flat /run for TOKENS without published rates", () => {
    render(
      <NodeCost
        blockCosts={[
          cost({ cost_type: BlockCostType.tokens, cost_amount: 150 }),
        ]}
        nodeId="n1"
      />,
    );
    expect(screen.getByText("$1.50")).toBeTruthy();
    expect(screen.getByText(/\/run/)).toBeTruthy();
  });

  it("renders 'Free' for RUN cost type with zero amount", () => {
    render(
      <NodeCost
        blockCosts={[cost({ cost_type: BlockCostType.run, cost_amount: 0 })]}
        nodeId="n1"
      />,
    );
    expect(screen.getByText("Free")).toBeTruthy();
    expect(screen.queryByText(/\$/)).toBeNull();
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
    } as Partial<ReturnType<typeof useNodeStore.getState>>);
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
