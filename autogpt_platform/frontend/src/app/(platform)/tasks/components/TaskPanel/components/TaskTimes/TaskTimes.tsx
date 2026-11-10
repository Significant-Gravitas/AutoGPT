import type { Task } from "@/app/api/__generated__/models/task";
import { Text } from "@/components/atoms/Text/Text";
import { timeRows } from "./helpers";

interface Props {
  task: Task;
}

/** When the task was created, started and finished. */
export function TaskTimes({ task }: Props) {
  return (
    <dl className="grid grid-cols-[auto_minmax(0,1fr)] gap-x-4 gap-y-1.5">
      {timeRows(task).map((row) => (
        <div key={row.label} className="contents">
          <Text as="dt" variant="inbox" tone="quiet">
            {row.label}
          </Text>
          <Text as="dd" variant="inbox" tone="text" tabular>
            {row.value ? (
              <time dateTime={row.value.toISOString()}>{row.formatted}</time>
            ) : (
              "Not yet"
            )}
          </Text>
        </div>
      ))}
    </dl>
  );
}
