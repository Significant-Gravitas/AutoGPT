import { Robot01Icon } from "@hugeicons/core-free-icons";
import {
  Avatar,
  AvatarFallback,
  AvatarImage,
} from "@/components/atoms/Avatar/Avatar";
import { Icon } from "@/components/atoms/Icon/Icon";
import type { AvatarStatus } from "@/components/molecules/NotionAvatar/expressions";
import { expertNotionConfig } from "@/components/molecules/NotionAvatar/helpers";
import { NotionAvatar } from "@/components/molecules/NotionAvatar/NotionAvatar";
import { NotionAvatarImage } from "@/components/molecules/NotionAvatar/NotionAvatarImage";
import { cn } from "@/lib/utils";

interface Props {
  name: string | null;
  avatarUrl: string | null;
  color?: string | null;
  status?: AvatarStatus;
  animated?: boolean;
  size?: number;
  className?: string;
}

/**
 * Expert avatar shared by the copilot home surfaces (briefing card, team
 * strip, needs-attention list). Uploaded pictures render as-is; everything
 * else gets the generated Notion-style face — as a flat image unless there
 * is something to animate.
 */
export function ExpertAvatar({
  name,
  avatarUrl,
  color,
  status = "idle",
  animated = false,
  size = 40,
  className,
}: Props) {
  const style = { width: size, height: size };

  if (!name) {
    return (
      <div
        style={style}
        className={cn(
          "flex shrink-0 items-center justify-center rounded-full bg-zinc-100",
          className,
        )}
      >
        <Icon icon={Robot01Icon} size={size / 2} className="text-zinc-500" />
      </div>
    );
  }

  const config = expertNotionConfig({ name, avatarUrl, color });
  if (config) {
    const isStill = !animated && status === "idle";
    return isStill ? (
      <NotionAvatarImage
        config={config}
        size={size}
        title={name}
        className={className}
      />
    ) : (
      <NotionAvatar
        config={config}
        status={status}
        size={size}
        animated={animated}
        showBadge={status !== "idle"}
        title={name}
        className={className}
      />
    );
  }

  return (
    <Avatar
      style={style}
      className={cn("shrink-0 border border-stone-500", className)}
    >
      <AvatarImage src={avatarUrl ?? undefined} alt={name} />
      <AvatarFallback>{name}</AvatarFallback>
    </Avatar>
  );
}
