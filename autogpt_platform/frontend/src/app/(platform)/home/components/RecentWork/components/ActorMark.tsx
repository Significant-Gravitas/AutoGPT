import type { HomeWorkActor } from "@/app/api/__generated__/models/homeWorkActor";
import { Icon } from "@/components/atoms/Icon/Icon";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import { getActorIcon } from "../helpers";

interface Props {
  actor: HomeWorkActor;
}

export function ActorMark({ actor }: Props) {
  if (actor.kind === "expert" && actor.expert) {
    return (
      <ExpertAvatar
        name={actor.expert.name}
        avatarUrl={actor.expert.avatar_url}
        size={18}
      />
    );
  }
  return (
    <span className="flex size-[18px] shrink-0 items-center justify-center rounded-full bg-zinc-100 text-zinc-500">
      <Icon icon={getActorIcon(actor.kind)} size={11} aria-hidden="true" />
    </span>
  );
}
