import { useMemo } from "react";
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
      // TODO(tokens): no exact role: bg-orange-200 #FFC88A
      "bg-orange-200",
      // TODO(tokens): no exact role: bg-red-200 #FAC5C5
      "bg-red-200",
      // TODO(tokens): no exact role: bg-yellow-200 #FAE28A
      "bg-yellow-200",
      // TODO(tokens): no exact role: bg-green-200 #94D5AC
      "bg-green-200",
      // TODO(tokens): no exact role: bg-blue-200 #BFDBFE
      "bg-blue-200",
      // TODO(tokens): no exact role: bg-purple-200 #C0A1FA
      "bg-purple-200",
      // TODO(tokens): no exact role: bg-pink-200 #F6ABD0
      "bg-pink-200",
    ];

    return {
      angle: (hash % 7) - 3,
      color: colors[Math.abs(hash) % colors.length],
    };
  }, [nodeId]);

  return (
    <div
      className={cn(
        // TODO(tokens): no exact role: shadow-[…] #000000
        "relative h-76 w-76 p-4 text-foreground-base shadow-[-2px_5px_5px_0px_rgba(0,0,0,0.3)]",
        color,
      )}
      style={{ transform: `rotate(${angle}deg)` }}
    >
      <Text
        // TODO(tokens): no exact role: text-slate-800 #475263
        variant="h3-tracking-tight"
        className="text-slate-800"
      >
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
