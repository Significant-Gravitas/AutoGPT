import {
  Avatar,
  AvatarFallback,
  AvatarImage,
} from "@/components/atoms/Avatar/Avatar";
import { AutopilotAvatar } from "@/components/molecules/AutopilotAvatar/AutopilotAvatar";

interface Props {
  name: string;
  avatarUrl: string | null;
  isAutopilot?: boolean;
  size?: "sm" | "md";
}

export function ExpertAvatar({ name, avatarUrl, isAutopilot, size }: Props) {
  const isSmall = size === "sm";
  const sizeClass = isSmall ? "h-6 w-6" : "h-9 w-9";

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
