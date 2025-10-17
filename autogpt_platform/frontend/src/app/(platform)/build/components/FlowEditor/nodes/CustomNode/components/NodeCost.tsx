import { BlockCost } from "@/app/api/__generated__/models/blockCost";
import { Text } from "@/components/atoms/Text/Text";
import useCredits from "@/hooks/useCredits";
import { CoinIcon } from "@phosphor-icons/react";
import { isCostFilterMatch } from "../../../../helper";
import { useNodeStore } from "@/app/(platform)/build/stores/nodeStore";

export const NodeCost = ({
  blockCosts,
  nodeId,
}: {
  blockCosts: BlockCost[];
  nodeId: string;
}) => {
  const { formatCredits } = useCredits();
  const hardcodedValues = useNodeStore((state) =>
    state.getHardCodedValues(nodeId),
  );
  const blockCost =
    blockCosts &&
    blockCosts.find((cost) =>
      isCostFilterMatch(cost.cost_filter, hardcodedValues),
    );

  if (!blockCost) return null;

  return (
    <div className="mr-3 flex items-center gap-1 text-base font-light">
      <CoinIcon className="h-3 w-3" />
      <Text variant="small" className="!font-medium">
        {formatCredits(blockCost.cost_amount)}
      </Text>
      <Text variant="small">
        {" \/"}
        {blockCost.cost_type}
      </Text>
    </div>
  );
};
