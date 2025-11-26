import { useNodeStore } from "@/app/(platform)/build/stores/nodeStore";
import { Switch } from "@/components/atoms/Switch/Switch";
import { Text } from "@/components/atoms/Text/Text";

export const NodeAdvancedToggle = ({ nodeId }: { nodeId: string }) => {
  const showAdvanced = useNodeStore(
    (state) => state.nodeAdvancedStates[nodeId] || false,
  );
  const setShowAdvanced = useNodeStore((state) => state.setShowAdvanced);
  return (
    <div className="flex items-center justify-between gap-2 rounded-b-xlarge border-t border-slate-200/50 bg-white px-5 py-3.5">
      <Text variant="body" className="font-medium text-slate-700">
        Advanced
      </Text>
      <Switch
        onCheckedChange={(checked) => setShowAdvanced(nodeId, checked)}
        checked={showAdvanced}
      />
    </div>
  );
};
