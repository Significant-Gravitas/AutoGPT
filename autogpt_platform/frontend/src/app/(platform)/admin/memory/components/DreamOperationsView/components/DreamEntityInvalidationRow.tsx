import type { EntityInvalidationSummary } from "@/app/api/__generated__/models/entityInvalidationSummary";
import { Text } from "@/components/atoms/Text/Text";
import { shortenUuid } from "../helpers";

interface Props {
  item: EntityInvalidationSummary;
}

export function DreamEntityInvalidationRow({ item }: Props) {
  const touched = item.edges_touched ?? [];
  return (
    // oxlint-disable-next-line design-system/no-raw-colors -- rose has no role; error would change the hue (deferred)
    <li className="border-l-2 border-rose-200 py-1.5 pl-3">
      <Text variant="small-medium" className="break-words">
        {item.reason}
      </Text>
      <div className="mt-1 flex flex-wrap items-center gap-2 text-foreground-tertiary">
        <code className="font-mono text-tiny leading-inherit">
          entity: {shortenUuid(item.entity_uuid)}
        </code>
        <Text variant="small" tone="muted" as="span">
          {touched.length} edge{touched.length === 1 ? "" : "s"} touched
        </Text>
      </div>
      {touched.length > 0 ? (
        <div className="mt-1 flex flex-wrap gap-1">
          {touched.slice(0, 6).map((u) => (
            <code
              key={u}
              className="rounded bg-surface-muted px-1.5 py-0.5 font-mono text-tiny leading-inherit text-foreground-secondary"
            >
              {shortenUuid(u)}
            </code>
          ))}
          {touched.length > 6 ? (
            <Text variant="small" tone="muted" as="span">
              +{touched.length - 6} more
            </Text>
          ) : null}
        </div>
      ) : null}
    </li>
  );
}
