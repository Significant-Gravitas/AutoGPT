import {
  Avatar,
  AvatarFallback,
  AvatarImage,
} from "@/components/atoms/Avatar/Avatar";
import { AutopilotAvatar } from "@/components/molecules/AutopilotAvatar/AutopilotAvatar";
import { expertNotionConfig } from "@/components/molecules/NotionAvatar/helpers";
import { NotionAvatarImage } from "@/components/molecules/NotionAvatar/NotionAvatarImage";
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
  const config = expertNotionConfig(identity);
  if (config) {
    return (
      <NotionAvatarImage
        config={config}
        size={imageSize}
        title={identity.name}
        className={className}
      />
    );
  }
  return (
    <Avatar className={cn("shrink-0 border border-stone-500", className)}>
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
