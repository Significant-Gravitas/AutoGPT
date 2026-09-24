import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import { AUTOPILOT_AVATAR_URL, AUTOPILOT_NAME } from "./helpers";

interface Props {
  size?: number;
  className?: string;
}

export function AutopilotAvatar({ size = 24, className }: Props) {
  return (
    <ExpertAvatar
      name={AUTOPILOT_NAME}
      avatarUrl={AUTOPILOT_AVATAR_URL}
      size={size}
      className={className}
    />
  );
}
