import { useNodeStore } from "@/app/(platform)/build/stores/nodeStore";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { CaretDownIcon } from "@phosphor-icons/react";

type Props = {
  nodeId: string;
};

export function NodeAdvancedToggle({ nodeId }: Props) {
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
        aria-expanded={showAdvanced}
      >
        <Text
          variant="body"
          as="span"
          className="flex items-center gap-2 !font-semibold text-slate-700"
        >
          Advanced{" "}
          <CaretDownIcon
            size={16}
            weight="bold"
            className={`transition-transform ${showAdvanced ? "rotate-180" : ""}`}
            aria-hidden
          />
        </Text>
      </Button>
    </div>
  );
}
