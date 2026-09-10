import {
  Avatar,
  AvatarFallback,
  AvatarImage,
} from "@/components/atoms/Avatar/Avatar";
import { cn } from "@/lib/utils";
import {
  AUTOPILOT_AVATAR_BG_CLASS,
  AUTOPILOT_AVATAR_URL,
  AUTOPILOT_NAME,
} from "./helpers";

interface Props {
  size?: number;
  className?: string;
}

/** Otto's face at any pixel size, on the cyan disc with the outline every
 *  expert avatar wears. */
export function AutopilotAvatar({ size = 24, className }: Props) {
  return (
    <Avatar
      style={{ width: size, height: size }}
      className={cn(
        "shrink-0 border border-stone-600",
        AUTOPILOT_AVATAR_BG_CLASS,
        className,
      )}
    >
      <AvatarImage
        src={AUTOPILOT_AVATAR_URL}
        alt={AUTOPILOT_NAME}
        width={size}
        height={size}
      />
      <AvatarFallback>{AUTOPILOT_NAME}</AvatarFallback>
    </Avatar>
  );
}
