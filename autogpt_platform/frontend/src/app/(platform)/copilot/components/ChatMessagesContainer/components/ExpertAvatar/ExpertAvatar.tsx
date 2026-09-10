import {
  Avatar,
  AvatarFallback,
  AvatarImage,
} from "@/components/atoms/Avatar/Avatar";
import { AutopilotAvatar } from "@/components/molecules/AutopilotAvatar/AutopilotAvatar";
import { cn } from "@/lib/utils";

interface Props {
  name: string;
  avatarUrl: string | null;
  isAutopilot?: boolean;
  /** Identity still loading: a muted grey circle, no stand-in identity. */
  isLoading?: boolean;
  size?: "sm" | "md";
}

export function ExpertAvatar({
  name,
  avatarUrl,
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
      <AutopilotAvatar
        size={isSmall ? 24 : 36}
        className={isSmall ? undefined : "rounded-xl"}
      />
    );
  }

  return (
    <Avatar className={sizeClass}>
      {avatarUrl ? <AvatarImage src={avatarUrl} alt={name} /> : null}
      {/* The fallback seeds a generated avatar off the name, and also covers
          an avatar URL that fails to load. */}
      <AvatarFallback className={sizeClass}>{name}</AvatarFallback>
    </Avatar>
  );
}
