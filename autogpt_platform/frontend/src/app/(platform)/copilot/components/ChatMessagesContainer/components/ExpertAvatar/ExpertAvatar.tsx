import {
  Avatar,
  AvatarFallback,
  AvatarImage,
} from "@/components/atoms/Avatar/Avatar";
import { BotAvatar } from "@/components/molecules/BotAvatar/BotAvatar";
import {
  AUTOPILOT_AVATAR,
  expertAvatarConfig,
  isUploadedAvatar,
} from "@/components/molecules/BotAvatar/helpers";
import { cn } from "@/lib/utils";

interface Props {
  name: string;
  avatarUrl: string | null;
  color?: string | null;
  isAutopilot?: boolean;
  /** Identity still loading: a muted grey circle, no stand-in identity. */
  isLoading?: boolean;
  size?: "sm" | "md";
}

export function ExpertAvatar({
  name,
  avatarUrl,
  color,
  isAutopilot,
  isLoading,
  size,
}: Props) {
  const isSmall = size === "sm";
  const sizeClass = isSmall ? "h-6 w-6" : "h-9 w-9";

  if (isLoading) {
    return (
      <span
        aria-hidden="true"
        className={cn("shrink-0 rounded-full bg-zinc-100", sizeClass)}
      />
    );
  }

  if (isAutopilot && !avatarUrl) {
    return (
      <BotAvatar
        config={AUTOPILOT_AVATAR}
        size={isSmall ? 24 : 36}
        animated={!isSmall}
        showBadge={false}
        title={name}
      />
    );
  }

  if (!isUploadedAvatar(avatarUrl)) {
    return (
      <BotAvatar
        config={expertAvatarConfig({ name, avatarUrl, color })}
        size={isSmall ? 24 : 36}
        animated={!isSmall}
        showBadge={false}
        title={name}
      />
    );
  }

  return (
    <Avatar className={sizeClass}>
      <AvatarImage src={avatarUrl ?? undefined} alt={name} />
      <AvatarFallback className={sizeClass}>{name}</AvatarFallback>
    </Avatar>
  );
}
