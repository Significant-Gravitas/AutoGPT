import { Robot01Icon } from "@hugeicons/core-free-icons";
import type { HomeWorkActor } from "@/app/api/__generated__/models/homeWorkActor";
import { Icon } from "@/components/atoms/Icon/Icon";
import { AutopilotAvatar } from "@/components/molecules/AutopilotAvatar/AutopilotAvatar";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import { WorkflowAvatar } from "@/components/molecules/WorkflowAvatar/WorkflowAvatar";

interface Props {
  actor: HomeWorkActor;
  size?: number;
  className?: string;
}

export function ActorMark({ actor, size = 18, className }: Props) {
  if (actor.kind === "expert" && actor.expert) {
    return (
      <ExpertAvatar
        name={actor.expert.name}
        avatarUrl={actor.expert.avatar_url}
        size={size}
        className={className}
      />
    );
  }
  if (actor.kind === "workflow") {
    return (
      <WorkflowAvatar
        name={actor.name}
        imageUrl={actor.image_url}
        size={18}
        className={className}
      />
    );
  }
  if (actor.kind === "autopilot") {
    return <AutopilotAvatar size={size} className={className} />;
  }
  return (
    <span className="flex size-[18px] shrink-0 items-center justify-center rounded-full bg-zinc-100 text-zinc-500">
      <Icon icon={Robot01Icon} size={11} aria-hidden="true" />
    </span>
  );
}
