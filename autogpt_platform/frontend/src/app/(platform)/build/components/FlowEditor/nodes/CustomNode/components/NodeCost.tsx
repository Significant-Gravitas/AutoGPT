import { BlockCost } from "@/app/api/__generated__/models/blockCost";
import { Text } from "@/components/atoms/Text/Text";
import useCredits from "@/hooks/useCredits";
import { CoinIcon } from "@phosphor-icons/react";
import { isCostFilterMatch } from "../../../../helper";
import { useNodeStore } from "@/app/(platform)/build/stores/nodeStore";
import { useShallow } from "zustand/react/shallow";

// Dynamic cost types bill off execution stats (walltime, tokens, provider
// USD) that aren't known pre-flight. For these the static cost_amount is
// either a floor (TOKENS) or zero (SECOND/ITEMS/COST_USD) — we render the
// unit instead of a single fixed number so users understand how billing
// scales. Matches backend BlockCostType variants in backend/blocks/_base.py.
type CostLabelKind = "fixed" | "per-unit" | "dynamic";

interface CostLabel {
  kind: CostLabelKind;
  unitSuffix: string;
  note?: string;
}

function getCostLabel(blockCost: BlockCost): CostLabel {
  const costType = blockCost.cost_type as string;
  const divisor = (blockCost as unknown as { cost_divisor?: number })
    .cost_divisor;
  switch (costType) {
    case "run":
      return { kind: "fixed", unitSuffix: "/run" };
    case "byte":
      return { kind: "per-unit", unitSuffix: "/byte" };
    case "second":
      return {
        kind: "per-unit",
        unitSuffix: divisor && divisor > 1 ? `/ ${divisor}s` : "/sec",
      };
    case "items":
      return {
        kind: "per-unit",
        unitSuffix: divisor && divisor > 1 ? `/ ${divisor} items` : "/item",
      };
    case "cost_usd":
      return {
        kind: "dynamic",
        unitSuffix: "/$",
        note: "Billed on provider-reported USD",
      };
    case "tokens":
      return {
        kind: "dynamic",
        unitSuffix: "/run",
        note: "Billed per-token (floor shown)",
      };
    default:
      return { kind: "fixed", unitSuffix: `/${costType}` };
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
