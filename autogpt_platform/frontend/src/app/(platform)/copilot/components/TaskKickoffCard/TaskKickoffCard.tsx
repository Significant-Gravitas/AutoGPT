import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { AttentionCard } from "@/components/organisms/AttentionCard/AttentionCard";
import { CheckListIcon } from "@hugeicons/core-free-icons";
import type { TaskKickoff } from "./helpers";

interface Props {
  kickoff: TaskKickoff;
}

export function TaskKickoffCard({ kickoff }: Props) {
  return (
    <AttentionCard.Root
      variant="outline"
      density="comfortable"
      className="max-w-md"
      data-testid="task-kickoff"
    >
      <AttentionCard.Row align="center">
        <AttentionCard.Icon>
          <Icon icon={CheckListIcon} size={18} />
        </AttentionCard.Icon>
        <AttentionCard.Content>
          <Text variant="small" tone="muted">
            Task {kickoff.taskKey}
          </Text>
          <Text variant="body-medium-tight" tone="primary" className="truncate">
            {kickoff.title}
          </Text>
        </AttentionCard.Content>
        <Button
          as="NextLink"
          href={`/tasks?task=${kickoff.taskId}`}
          variant="secondary"
          size="small"
        >
          Open task
        </Button>
      </AttentionCard.Row>
    </AttentionCard.Root>
  );
}
