import { Robot01Icon } from "@hugeicons/core-free-icons";
import {
  Avatar,
  AvatarFallback,
  AvatarImage,
} from "@/components/atoms/Avatar/Avatar";
import { Icon } from "@/components/atoms/Icon/Icon";
import { cn } from "@/lib/utils";

interface Props {
  name: string | null;
  avatarUrl: string | null;
  size?: number;
  className?: string;
}

/**
 * Expert avatar with a generic-agent fallback, shared by the copilot home
 * surfaces (briefing card, team strip, needs-attention list) so they stay
 * visually consistent.
 */
export function ExpertAvatar({ name, avatarUrl, size = 40, className }: Props) {
  const style = { width: size, height: size };

  if (!name) {
    return (
      <div
        style={style}
        className={cn(
          "flex shrink-0 items-center justify-center rounded-full bg-zinc-100",
          className,
        )}
      >
        <Icon icon={Robot01Icon} size={size / 2} className="text-zinc-500" />
      </div>
    );
  }

  return (
    <Avatar style={style} className={cn("shrink-0", className)}>
      {avatarUrl ? <AvatarImage src={avatarUrl} alt={name} /> : null}
      <AvatarFallback>{name}</AvatarFallback>
    </Avatar>
  );
}
