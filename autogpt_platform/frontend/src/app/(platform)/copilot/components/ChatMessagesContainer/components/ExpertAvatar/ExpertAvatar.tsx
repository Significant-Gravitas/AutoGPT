import { ExpertAvatar as SharedExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import { AutopilotAvatar } from "@/components/molecules/AutopilotAvatar/AutopilotAvatar";
import { cn } from "@/lib/utils";

interface Props {
  name: string;
  avatarUrl: string | null;
  color?: string | null;
  isAutopilot?: boolean;
  /** Identity still loading: a muted grey circle, no stand-in identity. */
  isLoading?: boolean;
  size?: "sm" | "md";
}

export function ExpertAvatar({
  name,
  avatarUrl,
  color,
  isAutopilot,
  isLoading,
  size,
}: Props) {
  const isSmall = size === "sm";
  const sizeClass = isSmall ? "h-6 w-6" : "h-9 w-9";
  // TODO(tokens): no exact role for border-[#e3e3e3]
  const borderClass = "rounded-full border border-[#e3e3e3]";

  if (isLoading) {
    return (
      <span
        aria-hidden="true"
        className={cn("shrink-0 rounded-full bg-surface-hover", sizeClass)}
      />
    );
  }

  if (isAutopilot && !avatarUrl) {
    return <AutopilotAvatar size={isSmall ? 24 : 36} className={borderClass} />;
  }

  return (
    <SharedExpertAvatar
      name={name}
      avatarUrl={avatarUrl}
      color={color}
      size={isSmall ? 24 : 36}
      className={borderClass}
    />
  );
}
