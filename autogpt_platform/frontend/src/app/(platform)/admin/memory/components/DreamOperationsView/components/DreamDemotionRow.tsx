import type { DemotionSummary } from "@/app/api/__generated__/models/demotionSummary";
import { Badge } from "@/components/atoms/Badge/Badge";
import { Text } from "@/components/atoms/Text/Text";
import { shortenUuid } from "../helpers";

interface Props {
  item: DemotionSummary;
}

export function DreamDemotionRow({ item }: Props) {
  return (
    <li className="border-l-2 border-amber-200 py-1.5 pl-3">
      <div className="flex items-start justify-between gap-2">
        <Text variant="small-medium" className="break-words">
          {item.reason}
        </Text>
        <Badge
          variant={item.applied ? "success" : "info"}
          size="small"
          className="whitespace-nowrap"
        >
          {item.new_status}
        </Badge>
      </div>
      <div className="mt-1 flex flex-wrap items-center gap-2 text-gray-500">
        <code className="font-mono text-[10px]">
          edge: {shortenUuid(item.edge_uuid)}
        </code>
        <Text variant="small" className="text-gray-500" as="span">
          {item.applied ? "applied" : "not applied"}
        </Text>
      </div>
    </li>
  );
}
