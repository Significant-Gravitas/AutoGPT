import type { WriteSummary } from "@/app/api/__generated__/models/writeSummary";
import { Badge } from "@/components/atoms/Badge/Badge";
import { Text } from "@/components/atoms/Text/Text";
import { formatConfidence, shortenUuid } from "../helpers";

interface Props {
  item: WriteSummary;
}

export function DreamWriteRow({ item }: Props) {
  return (
    <li className="border-l-2 border-violet-200 py-1.5 pl-3">
      <div className="flex items-start justify-between gap-2">
        <Text variant="small-medium" className="break-words">
          {item.content}
        </Text>
        {item.status ? (
          <Badge variant="info" size="small">
            {item.status}
          </Badge>
        ) : null}
      </div>
      <div className="mt-1 flex flex-wrap items-center gap-2 text-gray-500">
        <code className="font-mono text-[10px]">
          edge: {shortenUuid(item.edge_uuid)}
        </code>
        {item.scope ? (
          <Text variant="small" className="text-gray-500" as="span">
            scope: {item.scope}
          </Text>
        ) : null}
        <Text variant="small" className="text-gray-500" as="span">
          confidence: {formatConfidence(item.confidence)}
        </Text>
        {item.source_episode_uuids && item.source_episode_uuids.length > 0 ? (
          <Text variant="small" className="text-gray-500" as="span">
            sources: {item.source_episode_uuids.length}
          </Text>
        ) : null}
      </div>
    </li>
  );
}
