import { AutoGPTLogo } from "@/components/atoms/AutoGPTLogo/AutoGPTLogo";
import {
  Avatar,
  AvatarFallback,
  AvatarImage,
} from "@/components/atoms/Avatar/Avatar";
import { cn } from "@/lib/utils";

export interface PanelIdentity {
  name: string;
  avatarUrl: string | null;
  isAutopilot?: boolean;
}

interface Props {
  identity: PanelIdentity;
  className?: string;
  imageSize: number;
}

export function IdentityAvatar({ identity, className, imageSize }: Props) {
  if (identity.isAutopilot) {
    return (
      <span
        className={cn(
          "flex shrink-0 items-center justify-center rounded-full bg-white ring-1 ring-zinc-200",
          className,
        )}
      >
        <AutoGPTLogo hideText viewBox="47 -1 42 42" className="size-[55%]" />
      </span>
    );
  }
  return (
    <Avatar className={cn("shrink-0", className)}>
      {identity.avatarUrl ? (
        <AvatarImage
          src={identity.avatarUrl}
          alt={identity.name}
          width={imageSize}
          height={imageSize}
        />
      ) : null}
      <AvatarFallback>{identity.name}</AvatarFallback>
    </Avatar>
  );
}
