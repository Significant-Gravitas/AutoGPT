import type { TaskActivity } from "@/app/api/__generated__/models/taskActivity";
import { cn } from "@/lib/utils";
import { Dot } from "./TrailMarks";
import { TrailLineMeta } from "./TrailLineMeta";

interface Props {
  line: TaskActivity;
  isCurrent: boolean;
  isOpen: boolean;
}

/** One tool call made inside a step, on the branch under it. */
export function TrailSubLine({ line, isCurrent, isOpen }: Props) {
  return (
    <li
      title={line.text}
      className={cn(
        "flex h-6 -translate-y-1 items-center pr-2 pl-6 opacity-0 transition-[opacity,transform,translate] duration-200 ease-out",
        isOpen && "translate-y-0 opacity-100",
      )}
    >
      <span className="ml-[1.5px] flex">
        <Dot kind={isCurrent ? "current" : "done"} />
      </span>
      <span
        className={cn(
          "ml-[7.5px] min-w-0 truncate text-caption",
          isCurrent ? "text-foreground-text" : "text-foreground-quiet",
        )}
      >
        {line.text}
      </span>
      <TrailLineMeta line={line} />
    </li>
  );
}
