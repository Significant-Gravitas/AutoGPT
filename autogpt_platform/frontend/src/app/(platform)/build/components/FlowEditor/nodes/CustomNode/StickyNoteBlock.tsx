import { useMemo } from "react";
import { FormCreator } from "../FormCreator";
import { preprocessInputSchema } from "@/components/renderers/input-renderer/utils/input-schema-pre-processor";
import { CustomNodeData } from "./CustomNode";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";

type StickyNoteBlockType = {
  selected: boolean;
  data: CustomNodeData;
  id: string;
};

export const StickyNoteBlock = ({ data, id }: StickyNoteBlockType) => {
  const { angle, color } = useMemo(() => {
    const hash = id.split("").reduce((acc, char) => {
      return char.charCodeAt(0) + ((acc << 5) - acc);
    }, 0);

    const colors = [
      "bg-orange-200",
      "bg-red-200",
      "bg-yellow-200",
      "bg-green-200",
      "bg-blue-200",
      "bg-purple-200",
      "bg-pink-200",
    ];

    return {
      angle: (hash % 7) - 3,
      color: colors[Math.abs(hash) % colors.length],
    };
  }, [id]);

  return (
    <div
      className={cn(
        "relative h-76 w-76 p-4 text-black shadow-[rgba(0,0,0,0.3)_-2px_5px_5px_0px]",
        color,
      )}
      style={{ transform: `rotate(${angle}deg)` }}
    >
      <Text variant="h3" className="tracking-tight text-slate-800">
        Notes #{id.split("-")[0]}
      </Text>
      <FormCreator
        jsonSchema={preprocessInputSchema(data.inputSchema)}
        nodeId={id}
        uiType={data.uiType}
      />
    </div>
  );
};
