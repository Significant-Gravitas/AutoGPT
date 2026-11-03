import { CheckmarkCircle02Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Link } from "@/components/atoms/Link/Link";
import { Text } from "@/components/atoms/Text/Text";
import { AttentionCard } from "@/components/organisms/AttentionCard/AttentionCard";
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
    <AttentionCard.Root variant="success" density="snug">
      <AttentionCard.Row align="center">
        <AttentionCard.Icon>
          <Icon icon={CheckmarkCircle02Icon} size={16} tone="success" />
        </AttentionCard.Icon>
        <AttentionCard.Content>
          <Text variant="body-medium" as="span" tone="success-strong">
            {label}
          </Text>
          {goal ? (
            <Text variant="small" tone="secondary" className="truncate">
              {goal}
            </Text>
          ) : null}
        </AttentionCard.Content>
        {parentId ? (
          <Link
            href={taskHref(parentId)}
            className="shrink-0 rounded-sm text-body font-medium text-success-foreground focus-visible:ring-1 focus-visible:ring-focus focus-visible:outline-none"
          >
            Open task
          </Link>
        ) : null}
      </AttentionCard.Row>
    </AttentionCard.Root>
  );
}
