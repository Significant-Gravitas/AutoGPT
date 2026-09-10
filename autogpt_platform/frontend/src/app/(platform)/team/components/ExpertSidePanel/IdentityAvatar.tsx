import {
  Avatar,
  AvatarFallback,
  AvatarImage,
} from "@/components/atoms/Avatar/Avatar";
import { AutopilotAvatar } from "@/components/molecules/AutopilotAvatar/AutopilotAvatar";
import { BotAvatar } from "@/components/molecules/BotAvatar/BotAvatar";
import {
  expertAvatarConfig,
  isUploadedAvatar,
} from "@/components/molecules/BotAvatar/helpers";
import { cn } from "@/lib/utils";

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
  if (identity.isAutopilot) {
    return <AutopilotAvatar size={imageSize} className={className} />;
  }
  if (!isUploadedAvatar(identity.avatarUrl)) {
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
    <Avatar className={cn("shrink-0 border border-zinc-800", className)}>
      <AvatarImage
        src={identity.avatarUrl ?? undefined}
        alt={identity.name}
        width={imageSize}
        height={imageSize}
      />
      <AvatarFallback>{identity.name}</AvatarFallback>
    </Avatar>
  );
}
