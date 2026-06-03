import { BlockCost } from "@/app/api/__generated__/models/blockCost";
import { BlockCostType } from "@/app/api/__generated__/models/blockCostType";
import { Text } from "@/components/atoms/Text/Text";
import useCredits from "@/hooks/useCredits";
import { CoinIcon } from "@phosphor-icons/react";
import { isCostFilterMatch } from "../../../../helper";
import { useNodeStore } from "@/app/(platform)/build/stores/nodeStore";
import { useShallow } from "zustand/react/shallow";

type CostDisplay =
  | { kind: "free" }
  | {
      kind: "token-rate";
      input: number;
      output: number;
      cacheRead: number | null;
      cacheCreation: number | null;
    }
  | { kind: "by-usd" }
  | { kind: "fixed"; amountText: string; suffix: string; note?: string }
  | { kind: "per-unit"; amountText: string; suffix: string; note?: string };

function formatUsd(value: number): string {
  return `$${value.toFixed(2)}`;
}

function tokenRateTooltip(d: {
  input: number;
  output: number;
  cacheRead: number | null;
  cacheCreation: number | null;
}): string {
  const parts = [
    `Input: ${formatUsd(d.input)} / 1M tokens`,
    `Output: ${formatUsd(d.output)} / 1M tokens`,
  ];
  if (d.cacheRead != null)
    parts.push(`Cached input: ${formatUsd(d.cacheRead)} / 1M tokens`);
  if (d.cacheCreation != null)
    parts.push(`Cache write: ${formatUsd(d.cacheCreation)} / 1M tokens`);
  parts.push("Final charge scales with real usage (1.5× margin applied).");
  return parts.join(" · ");
}

function getDisplay(
  blockCost: BlockCost,
  formatCredits: (n: number | null) => string,
): CostDisplay {
  // Token-rate pair takes precedence regardless of cost_type so that
  // OpenRouter (COST_USD)-billed models show the same "$X / $Y per 1M"
  // format as direct-billed providers when we have a published rate.
  // The internal billing path (TOKENS vs COST_USD) is not user-facing.
  if (blockCost.token_rate) {
    return {
      kind: "token-rate",
      input: blockCost.token_rate.input_usd_per_1m,
      output: blockCost.token_rate.output_usd_per_1m,
      cacheRead: blockCost.token_rate.cache_read_usd_per_1m ?? null,
      cacheCreation: blockCost.token_rate.cache_creation_usd_per_1m ?? null,
    };
  }

  if (blockCost.cost_type === BlockCostType.cost_usd) {
    return { kind: "by-usd" };
  }

  if (
    blockCost.cost_type === BlockCostType.run &&
    blockCost.cost_amount === 0
  ) {
    return { kind: "free" };
  }

  const divisor = blockCost.cost_divisor;
  const amountText = formatCredits(blockCost.cost_amount);

  switch (blockCost.cost_type) {
    case BlockCostType.run:
      return { kind: "fixed", amountText, suffix: "/run" };
    case BlockCostType.byte:
      return { kind: "per-unit", amountText, suffix: "/byte" };
    case BlockCostType.second:
      return {
        kind: "per-unit",
        amountText,
        suffix: divisor && divisor > 1 ? `/ ${divisor}s` : "/sec",
      };
    case BlockCostType.items:
      return {
        kind: "per-unit",
        amountText,
        suffix: divisor && divisor > 1 ? `/ ${divisor} items` : "/item",
      };
    case BlockCostType.tokens:
      return {
        kind: "fixed",
        amountText,
        suffix: "/run",
        note: "Pre-flight flat estimate — per-token rate not in our catalog yet.",
      };
    default:
      return { kind: "fixed", amountText, suffix: `/${blockCost.cost_type}` };
  }
}

export const NodeCost = ({
  blockCosts,
  nodeId,
}: {
  blockCosts: BlockCost[];
  nodeId: string;
}) => {
  const { formatCredits } = useCredits();
  const hardcodedValues = useNodeStore(
    useShallow((state) => state.getHardCodedValues(nodeId)),
  );

  const blockCost =
    blockCosts &&
    blockCosts.find((cost) =>
      isCostFilterMatch(cost.cost_filter, hardcodedValues),
    );

  if (!blockCost) return null;

  const display = getDisplay(blockCost, formatCredits);

  if (display.kind === "free") {
    return (
      <div className="mr-3 flex items-center gap-1 text-base font-light">
        <CoinIcon className="h-3 w-3" />
        <Text variant="small" className="!font-medium">
          Free
        </Text>
      </div>
    );
  }

  if (display.kind === "token-rate") {
    return (
      <div
        className="mr-3 flex items-center gap-1 text-base font-light"
        title={tokenRateTooltip(display)}
      >
        <CoinIcon className="h-3 w-3" />
        <Text variant="small" className="!font-medium">
          {`${formatUsd(display.input)} in / ${formatUsd(display.output)} out`}
        </Text>
        <Text variant="small">{" per 1M tokens"}</Text>
      </div>
    );
  }

  if (display.kind === "by-usd") {
    return (
      <div
        className="mr-3 flex items-center gap-1 text-base font-light"
        title="Pay-as-you-go pricing. Final charge matches the provider's per-token rate plus a 1.5× margin."
      >
        <CoinIcon className="h-3 w-3" />
        <Text variant="small">Pay-as-you-go</Text>
      </div>
    );
  }

  return (
    <div
      className="mr-3 flex items-center gap-1 text-base font-light"
      title={display.note}
    >
      <CoinIcon className="h-3 w-3" />
      <Text variant="small" className="!font-medium">
        {display.amountText}
      </Text>
      <Text variant="small">{` ${display.suffix}`}</Text>
    </div>
  );
};
