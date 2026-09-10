"use client";

import { Expert } from "@/app/api/__generated__/models/expert";
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

interface Props {
  expert: Pick<Expert, "name" | "avatar_url" | "color">;
  size: number;
  className?: string;
}

/** A generated or missing avatar renders as the live BotAvatar so the face
 *  blinks and follows the pointer like everywhere else; only a real upload
 *  falls back to a static image. */
export function ExpertFace({ expert, size, className }: Props) {
  if (isUploadedAvatar(expert.avatar_url)) {
    return (
      <Avatar
        className={cn("bg-white shadow-sm ring-1 ring-black/5", className)}
      >
        <AvatarImage src={expert.avatar_url ?? undefined} alt={expert.name} />
        <AvatarFallback>{expert.name}</AvatarFallback>
      </Avatar>
    );
  }
  return (
    <BotAvatar
      config={expertAvatarConfig({
        name: expert.name,
        avatarUrl: expert.avatar_url,
        color: expert.color,
      })}
      size={size}
      trackPointer
      showBadge={false}
      title={expert.name}
      className={cn("shrink-0", className)}
    />
  );
}
