import { Expert } from "@/app/api/__generated__/models/expert";
import {
  Avatar,
  AvatarFallback,
  AvatarImage,
} from "@/components/atoms/Avatar/Avatar";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { PlusIcon } from "@phosphor-icons/react";

interface Props {
  expert: Expert;
  onInstallWorkflow: (expertId: string) => void;
}

export function ExpertTeamCard({ expert, onInstallWorkflow }: Props) {
  const workflowCount = expert.workflows.length;

  return (
    <div className="flex flex-col gap-3 rounded-2xl border border-zinc-200 bg-white p-5">
      <div className="flex items-center gap-3">
        <Avatar className="h-12 w-12">
          {expert.avatar_url ? (
            <AvatarImage src={expert.avatar_url} alt={expert.name} />
          ) : null}
          <AvatarFallback>{expert.name}</AvatarFallback>
        </Avatar>
        <div className="min-w-0 flex-1">
          <Text variant="large-medium">{expert.name}</Text>
          <Text variant="small" className="text-zinc-500">
            {expert.role}
          </Text>
        </div>
      </div>
      <div className="flex flex-col gap-2">
        <Text variant="small" className="text-zinc-500">
          {workflowCount} {workflowCount === 1 ? "workflow" : "workflows"}
        </Text>
        {workflowCount > 0 ? (
          <div className="flex flex-wrap gap-1">
            {expert.workflows.map((workflow) =>
              workflow.name ? (
                <span
                  key={workflow.id}
                  className="rounded-full bg-zinc-100 px-2 py-0.5 text-xs text-zinc-700"
                >
                  {workflow.name}
                </span>
              ) : null,
            )}
          </div>
        ) : null}
      </div>
      <div className="mt-auto flex gap-2">
        <Button
          as="NextLink"
          href={`/copilot?expertId=${expert.id}`}
          variant="secondary"
          size="small"
        >
          Chat
        </Button>
        <Button
          variant="ghost"
          size="small"
          leftIcon={<PlusIcon size={16} />}
          onClick={() => onInstallWorkflow(expert.id)}
        >
          Install workflow
        </Button>
      </div>
    </div>
  );
}
