import {
  Avatar,
  AvatarFallback,
  AvatarImage,
} from "@/components/atoms/Avatar/Avatar";
import { BotAvatar } from "@/components/molecules/BotAvatar/BotAvatar";
import {
  expertAvatarConfig,
  isUploadedAvatar,
} from "@/components/molecules/BotAvatar/helpers";
import { cn } from "@/lib/utils";
import { AUTOPILOT_AVATAR_BG_CLASS, AUTOPILOT_AVATAR_URL } from "../../helpers";

export interface PanelIdentity {
  name: string;
  avatarUrl: string | null;
  color?: string | null;
  isAutopilot?: boolean;
}

interface Props {
  identity: PanelIdentity;
  className?: string;
  imageSize: number;
}

export function IdentityAvatar({ identity, className, imageSize }: Props) {
  const avatarUrl = identity.isAutopilot
    ? AUTOPILOT_AVATAR_URL
    : identity.avatarUrl;
  if (!isUploadedAvatar(avatarUrl)) {
    return (
      <BotAvatar
        config={expertAvatarConfig(identity)}
        size={imageSize}
        showBadge={false}
        title={identity.name}
        className={className}
      />
    );
  }
  return (
    <Avatar className={cn("shrink-0", className)}>
      <AvatarImage
        src={avatarUrl ?? undefined}
        alt={identity.name}
        width={imageSize}
        height={imageSize}
        className={cn(identity.isAutopilot && AUTOPILOT_AVATAR_BG_CLASS)}
      />
      <AvatarFallback>{identity.name}</AvatarFallback>
    </Avatar>
  );
}
