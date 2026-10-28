// TODO(tokens): no role for these colours yet, so they stay raw:
// bg-gray-50 #F9FAFB, border-rose-200 #FECDD3, text-gray-500 #6B7280,
// text-gray-600 #4B5563.

import type { EntityInvalidationSummary } from "@/app/api/__generated__/models/entityInvalidationSummary";
import { Text } from "@/components/atoms/Text/Text";
import { shortenUuid } from "../helpers";

interface Props {
  item: EntityInvalidationSummary;
}

export function DreamEntityInvalidationRow({ item }: Props) {
  const touched = item.edges_touched ?? [];
  return (
    <li className="border-l-2 border-rose-200 py-1.5 pl-3">
      <Text variant="small-medium" className="break-words">
        {item.reason}
      </Text>
      <div className="mt-1 flex flex-wrap items-center gap-2 text-gray-500">
        <code className="font-mono text-tiny leading-inherit">
          entity: {shortenUuid(item.entity_uuid)}
        </code>
        <Text variant="small" className="text-gray-500" as="span">
          {touched.length} edge{touched.length === 1 ? "" : "s"} touched
        </Text>
      </div>
      {touched.length > 0 ? (
        <div className="mt-1 flex flex-wrap gap-1">
          {touched.slice(0, 6).map((u) => (
            <code
              key={u}
              className="rounded bg-gray-50 px-1.5 py-0.5 font-mono text-tiny leading-inherit text-gray-600"
            >
              {shortenUuid(u)}
            </code>
          ))}
          {touched.length > 6 ? (
            <Text variant="small" className="text-gray-500" as="span">
              +{touched.length - 6} more
            </Text>
          ) : null}
        </div>
      ) : null}
    </li>
  );
}
