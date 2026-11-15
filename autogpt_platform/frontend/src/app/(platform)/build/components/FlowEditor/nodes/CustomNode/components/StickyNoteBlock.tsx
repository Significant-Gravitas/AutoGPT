import { useMemo, type CSSProperties } from "react";
import { FormCreator } from "../../FormCreator";
import { preprocessInputSchema } from "@/components/renderers/InputRenderer/utils/input-schema-pre-processor";
import { CustomNodeData } from "../CustomNode";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";

type StickyNoteBlockType = {
  selected: boolean;
  data: CustomNodeData;
  nodeId: string;
};

export const StickyNoteBlock = ({ data, nodeId }: StickyNoteBlockType) => {
  const { angle, color } = useMemo(() => {
    const hash = nodeId.split("").reduce((acc, char) => {
      return char.charCodeAt(0) + ((acc << 5) - acc);
    }, 0);

    const colors = [
      "bg-[#ffc88a]",
      "bg-[#fac5c5]",
      "bg-[#fae28a]",
      "bg-[#94d5ac]",
      "bg-[#bfdbfe]",
      "bg-[#c0a1fa]",
      "bg-[#f6abd0]",
    ];

    return {
      angle: (hash % 7) - 3,
      color: colors[Math.abs(hash) % colors.length],
    };
  }, [nodeId]);

  return (
    <div
      className={cn(
        // oxlint-disable-next-line design-system/no-arbitrary-values -- the note's lifted-paper shadow (offset left and down) has no shadow token
        "relative h-76 w-76 rotate-[var(--note-angle)] p-4 text-foreground-base shadow-[-2px_5px_5px_0px_rgba(0,0,0,0.3)]",
        color,
      )}
      style={{ "--note-angle": `${angle}deg` } as CSSProperties}
    >
      <Text variant="h3-tracking-tight" tone="body">
        Notes #{nodeId.split("-")[0]}
      </Text>
      <FormCreator
        jsonSchema={preprocessInputSchema(data.inputSchema)}
        nodeId={nodeId}
        uiType={data.uiType}
      />
    </div>
  );
};
