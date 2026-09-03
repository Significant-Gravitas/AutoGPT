import { Chatting01Icon, EyeIcon } from "@hugeicons/core-free-icons";
import NextLink from "next/link";
import { StatusBadge } from "@/app/(platform)/library/components/StatusBadge/StatusBadge";
import type { SitrepItemData } from "@/app/(platform)/library/types";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { buildAskHref } from "../helpers";
import styles from "../RecentWorkflowRuns.module.css";

interface Props {
  run: SitrepItemData;
}

export function RunChip({ run }: Props) {
  return (
    <div
      className={`${styles.chip} relative flex w-[15rem] shrink-0 flex-col items-start gap-2 rounded-medium border border-zinc-200 bg-white px-3 py-2`}
    >
      <div className={`${styles.chipContent} w-full text-left`}>
        {run.priority === "success" ? (
          <span className="inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-xs font-medium text-emerald-600">
            <span className="h-1.5 w-1.5 rounded-full bg-emerald-500" />
            Completed
          </span>
        ) : (
          <StatusBadge status={run.status} />
        )}
        <div className="mt-2 min-w-0">
          <Text variant="small-medium" className="truncate text-zinc-900">
            {run.agentName}
          </Text>
          <Text variant="small" className="truncate text-zinc-500">
            {run.message}
          </Text>
        </div>
      </div>
      <div
        className={`${styles.chipActions} flex items-center justify-center gap-1.5 rounded-b-medium px-3 py-1.5`}
      >
        <NextLink
          href={`/library/agents/${run.agentID}`}
          className="flex items-center gap-1 rounded-md px-2 py-1 text-xs text-zinc-500 transition-colors hover:bg-zinc-100 hover:text-zinc-700"
        >
          <Icon icon={EyeIcon} size={14} />
          See
        </NextLink>
        <NextLink
          href={buildAskHref(run)}
          className="flex items-center gap-1 rounded-md px-2 py-1 text-xs text-zinc-500 transition-colors hover:bg-zinc-100 hover:text-zinc-700"
        >
          <Icon icon={Chatting01Icon} size={14} />
          Ask
        </NextLink>
      </div>
    </div>
  );
}
