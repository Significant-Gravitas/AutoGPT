import { useNodeStore } from "@/app/(platform)/build/stores/nodeStore";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { CaretDownIcon } from "@phosphor-icons/react";

export const NodeAdvancedToggle = ({ nodeId }: { nodeId: string }) => {
  const showAdvanced = useNodeStore(
    (state) => state.nodeAdvancedStates[nodeId] || false,
  );
  const setShowAdvanced = useNodeStore((state) => state.setShowAdvanced);
  return (
    <div className="flex items-center justify-start gap-2 bg-white px-5 pb-3.5">
      <Button
        variant="ghost"
        className="h-fit min-w-0 p-0 hover:border-transparent hover:bg-transparent"
        onClick={() => setShowAdvanced(nodeId, !showAdvanced)}
      >
        <Text
          variant="body"
          className="flex items-center gap-2 !font-semibold text-slate-700"
        >
          Advanced{" "}
          <CaretDownIcon
            size={16}
            weight="bold"
            className={`transition-transform ${showAdvanced ? "rotate-180" : ""}`}
          />
        </Text>
      </Button>
    </div>
  );
};
