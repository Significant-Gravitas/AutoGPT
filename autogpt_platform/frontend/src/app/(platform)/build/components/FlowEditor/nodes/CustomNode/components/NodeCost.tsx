import { BlockCost } from "@/app/api/__generated__/models/blockCost";
import { BlockCostType } from "@/app/api/__generated__/models/blockCostType";
import { Text } from "@/components/atoms/Text/Text";
import useCredits from "@/hooks/useCredits";
import { CoinIcon } from "@phosphor-icons/react";
import { isCostFilterMatch } from "../../../../helper";
import { useNodeStore } from "@/app/(platform)/build/stores/nodeStore";
import { useShallow } from "zustand/react/shallow";

type CostLabelKind = "fixed" | "per-unit" | "dynamic";

interface CostLabel {
  kind: CostLabelKind;
  unitSuffix: string;
  note?: string;
}

function getCostLabel(blockCost: BlockCost): CostLabel {
  const divisor = blockCost.cost_divisor;
  switch (blockCost.cost_type) {
    case BlockCostType.run:
      return { kind: "fixed", unitSuffix: "/run" };
    case BlockCostType.byte:
      return { kind: "per-unit", unitSuffix: "/byte" };
    case BlockCostType.second:
      return {
        kind: "per-unit",
        unitSuffix: divisor && divisor > 1 ? `/ ${divisor}s` : "/sec",
      };
    case BlockCostType.items:
      return {
        kind: "per-unit",
        unitSuffix: divisor && divisor > 1 ? `/ ${divisor} items` : "/item",
      };
    case BlockCostType.cost_usd:
      return {
        kind: "dynamic",
        unitSuffix: "· by USD",
        note: "Final charge scales with provider-reported USD spend",
      };
    case BlockCostType.tokens:
      return {
        kind: "dynamic",
        unitSuffix: "· by tokens",
        note: "Floor shown; final charge scales with real token usage (cached tokens discounted)",
      };
    default:
      return { kind: "fixed", unitSuffix: `/${blockCost.cost_type}` };
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

  const label = getCostLabel(blockCost);
  const amountText =
    label.kind === "fixed"
      ? formatCredits(blockCost.cost_amount)
      : blockCost.cost_amount > 0
        ? `~${formatCredits(blockCost.cost_amount)}`
        : "—";

  return (
    <div
      className="mr-3 flex items-center gap-1 text-base font-light"
      title={label.note}
    >
      <CoinIcon className="h-3 w-3" />
      <Text variant="small" className="!font-medium">
        {amountText}
      </Text>
      <Text variant="small">{` ${label.unitSuffix}`}</Text>
    </div>
  );
};
