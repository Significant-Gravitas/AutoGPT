import { Fragment } from "react";
import Link from "next/link";
import type { BriefingMergeItem } from "@/app/api/__generated__/models/briefingMergeItem";
import { Text } from "@/components/atoms/Text/Text";

interface Props {
  item: BriefingMergeItem;
}

// Two open tasks that look like the same ask — a suggestion only, so both
// titles link to their tasks and nothing is merged automatically.
export function MergeRow({ item }: Props) {
  return (
    <li className="px-5 py-4">
      <div className="flex flex-wrap items-center gap-x-1.5 gap-y-0.5">
        {item.titles.map((title, index) => {
          const taskId = item.task_ids[index];
          return (
            <Fragment key={taskId ?? index}>
              {index > 0 ? (
                <span aria-hidden className="text-zinc-300">
                  ·
                </span>
              ) : null}
              {taskId ? (
                <Link
                  href={`/team/tasks/${taskId}`}
                  className="truncate text-[0.9375rem] font-medium text-zinc-900 hover:underline"
                >
                  {title}
                </Link>
              ) : (
                <Text
                  variant="body-medium"
                  unmask={false}
                  className="truncate text-zinc-900"
                >
                  {title}
                </Text>
              )}
            </Fragment>
          );
        })}
      </div>
      <Text variant="body" className="mt-0.5 text-zinc-500">
        look like the same ask — merge?
      </Text>
    </li>
  );
}
