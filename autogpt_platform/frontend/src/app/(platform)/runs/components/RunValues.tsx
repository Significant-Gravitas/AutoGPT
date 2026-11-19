import { ValueView } from "@/app/(platform)/copilot/components/WorkArea/WorkflowRunTab/components/ValueView";
import {
  labelOf,
  unwrap,
  visiblePins,
} from "@/app/(platform)/copilot/components/WorkArea/WorkflowRunTab/value";
import { Text } from "@/components/atoms/Text/Text";

interface Props {
  value: Record<string, unknown> | null;
}

/** A run's or a block's input or output drawn to be read, as a table: the
 *  name on the left, the value (text, links, lists and fields) on the
 *  right. */
export function RunValues({ value }: Props) {
  const entries = visiblePins(value);
  if (entries.length === 0) {
    return (
      <Text variant="small" as="p" tone="faint">
        Nothing
      </Text>
    );
  }
  return (
    <dl className="divide-y divide-border-hairline overflow-hidden rounded-xl bg-surface shadow-(--inbox-shadow-card)">
      {entries.map(([key, item]) => (
        <div
          key={key}
          className="grid grid-cols-[minmax(4.5rem,30%)_minmax(0,1fr)] gap-4 px-3.5 py-2.5"
        >
          <Text variant="inbox" as="dt" tone="quiet" className="truncate">
            {labelOf(key)}
          </Text>
          <dd className="min-w-0">
            <ValueView value={unwrap(item)} />
          </dd>
        </div>
      ))}
    </dl>
  );
}
