import { Expert } from "@/app/api/__generated__/models/expert";
import {
  Avatar,
  AvatarFallback,
  AvatarImage,
} from "@/components/atoms/Avatar/Avatar";
import { Icon } from "@/components/atoms/Icon/Icon";
import { cn } from "@/lib/utils";
import { ExpertAccent, getExpertAvatarUrl } from "../../../../helpers";

type Props = {
  expert: Expert;
  accent: ExpertAccent;
};

export function ExpertProfileHeader({ expert, accent }: Props) {
  const avatarUrl = getExpertAvatarUrl(expert);

  return (
    <div
      className={cn(
        "relative flex items-center gap-5 overflow-hidden rounded-2xl border border-zinc-200/60 p-5",
        accent.wash,
      )}
    >
      <Avatar className="h-24 w-24 bg-white shadow-sm ring-1 ring-black/5">
        {avatarUrl ? <AvatarImage src={avatarUrl} alt={expert.name} /> : null}
        <AvatarFallback>{expert.name}</AvatarFallback>
      </Avatar>
      <div>
        <div className="flex items-center gap-3">
          <h2 className="text-3xl font-semibold tracking-[-0.02em] text-zinc-900">
            {expert.name}
          </h2>
          <span
            className={cn(
              "inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-sm font-medium",
              accent.pill,
            )}
          >
            <Icon icon={accent.roleIcon} size={14} />
            {expert.role}
          </span>
        </div>
        {expert.tagline ? (
          <p className="mt-1.5 text-base text-zinc-500">{expert.tagline}</p>
        ) : null}
      </div>
    </div>
  );
}
