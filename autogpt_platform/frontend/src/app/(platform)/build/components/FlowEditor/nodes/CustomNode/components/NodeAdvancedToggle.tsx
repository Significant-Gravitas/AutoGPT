import { useNodeStore } from "@/app/(platform)/build/stores/nodeStore";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { ArrowDown01Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

type Props = {
  nodeId: string;
  isLastSection?: boolean;
  hasAdvancedFields?: boolean;
};

export function NodeAdvancedToggle({
  nodeId,
  isLastSection,
  hasAdvancedFields = true,
}: Props) {
  const showAdvanced = useNodeStore(
    (state) => state.nodeAdvancedStates[nodeId] || false,
  );
  const setShowAdvanced = useNodeStore((state) => state.setShowAdvanced);

  if (!hasAdvancedFields) return null;

  return (
    <div
      className={cn(
        "flex items-center justify-start gap-2 bg-surface px-5 pb-3.5",
        isLastSection && "rounded-b-xlarge",
      )}
    >
      <Button
        variant="toggle"
        size="inline"
        onClick={() => setShowAdvanced(nodeId, !showAdvanced)}
        aria-expanded={showAdvanced}
      >
        <Text
          variant="body-semibold"
          tone="body"
          as="span"
          className="flex items-center gap-2"
        >
          Advanced{" "}
          <Icon
            icon={ArrowDown01Icon}
            size={16}
            motion="transform"
            className={showAdvanced ? "rotate-180" : undefined}
            aria-hidden
          />
        </Text>
      </Button>
    </div>
  );
}
