import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import {
  AVATAR_SHAPES,
  type AvatarShape,
} from "@/components/molecules/ExpertAvatar/helpers";
import { cn } from "@/lib/utils";
import Image from "next/image";
import {
  AUTOPILOT_AVATAR_URL,
  AUTOPILOT_NAME,
  AUTOPILOT_TRANSPARENT_AVATAR_URL,
} from "./helpers";

interface Props {
  size?: number;
  transparent?: boolean;
  /** Corner radius over the avatar's own (see AVATAR_SHAPES). */
  shape?: AvatarShape;
  className?: string;
  backgroundColor?: string;
}

export function AutopilotAvatar({
  size = 24,
  transparent = false,
  shape,
  className,
  backgroundColor,
}: Props) {
  if (transparent) {
    return (
      <Image
        src={AUTOPILOT_TRANSPARENT_AVATAR_URL}
        alt={`${AUTOPILOT_NAME}, your personal Head of AI`}
        width={size}
        height={size}
        sizes={`${size}px`}
        data-slot="autopilot-avatar"
        className={cn(
          "shrink-0 object-contain",
          shape && AVATAR_SHAPES[shape],
          className,
        )}
      />
    );
  }

  return (
    <ExpertAvatar
      name={AUTOPILOT_NAME}
      avatarUrl={AUTOPILOT_AVATAR_URL}
      size={size}
      backgroundColor={backgroundColor}
      shape={shape}
      className={className}
    />
  );
}
