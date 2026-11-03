import { CheckmarkCircle02Icon } from "@hugeicons/core-free-icons";
import NextLink from "next/link";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { taskKey } from "@/app/(platform)/tasks/data/helpers";
import { taskHref } from "../helpers";

interface Props {
  goal?: string | null;
  parentId: string | null;
  parentNumber: number | null;
}

export function PlanStartedCard({ goal, parentId, parentNumber }: Props) {
  const label = parentNumber
    ? `Plan started · ${taskKey(parentNumber)}`
    : "Plan started";
  return (
    <div className="flex animate-inbox-enter items-center gap-2.5 rounded-card bg-status-ok-surface px-3.5 py-3">
      <Icon icon={CheckmarkCircle02Icon} size={16} tone="ok" aria-hidden />
      <div className="flex min-w-0 flex-1 flex-col">
        <Text variant="inbox-medium" as="span" tone="ok">
          {label}
        </Text>
        {goal ? (
          <Text variant="inbox" tone="text" className="truncate">
            {goal}
          </Text>
        ) : null}
      </div>
      {parentId ? (
        <NextLink
          href={taskHref(parentId)}
          className="shrink-0 rounded-control px-2 py-1 transition-[background-color] duration-150 ease-out-expo hover:bg-surface-wash focus-visible:ring-2 focus-visible:ring-focus-inbox focus-visible:outline-none"
        >
          <Text variant="inbox-medium" as="span" tone="ok">
            Open task
          </Text>
        </NextLink>
      ) : null}
    </div>
  );
}
