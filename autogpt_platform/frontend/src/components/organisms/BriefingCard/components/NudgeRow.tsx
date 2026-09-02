import Link from "next/link";
import { ArrowRight01Icon } from "@hugeicons/core-free-icons";
import type { BriefingNudgeItem } from "@/app/api/__generated__/models/briefingNudgeItem";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";

interface Props {
  item: BriefingNudgeItem;
}

// A WAITING_USER task the user has sat on: the question doubles as the row's
// subtitle, mirroring RunRow's single-line title + detail.
export function NudgeRow({ item }: Props) {
  return (
    <li>
      <Link
        href={`/team/tasks/${item.task_id}`}
        className="group flex items-start gap-3 px-5 py-4 transition-colors hover:bg-zinc-50"
      >
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <Text
              variant="body-medium"
              unmask={false}
              className="truncate text-zinc-900"
            >
              {item.title}
            </Text>
            {item.is_stale ? (
              <span className="shrink-0 rounded-full bg-amber-50 px-2 py-0.5 text-[0.6875rem] font-medium text-amber-700">
                Stale
              </span>
            ) : null}
          </div>
          {item.question ? (
            <Text
              variant="body"
              unmask={false}
              className="line-clamp-2 text-zinc-500"
            >
              {item.question}
            </Text>
          ) : null}
        </div>
        <Icon
          icon={ArrowRight01Icon}
          size={16}
          className="shrink-0 self-center text-zinc-300 transition-colors group-hover:text-zinc-500"
        />
      </Link>
    </li>
  );
}
