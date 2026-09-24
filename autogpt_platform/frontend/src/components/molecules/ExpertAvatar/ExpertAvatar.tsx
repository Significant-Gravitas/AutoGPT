import { Robot01Icon } from "@hugeicons/core-free-icons";
import {
  Avatar,
  AvatarFallback,
  AvatarImage,
} from "@/components/atoms/Avatar/Avatar";
import { Icon } from "@/components/atoms/Icon/Icon";
import type { AvatarStatus } from "@/components/molecules/NotionAvatar/status";
import { expertNotionConfig } from "@/components/molecules/NotionAvatar/helpers";
import { NotionAvatarImage } from "@/components/molecules/NotionAvatar/NotionAvatarImage";
import { cn } from "@/lib/utils";
import { getManagedAvatar } from "./helpers";
import { ManagedExpertImage } from "./components/ManagedExpertImage";

interface Props {
  name: string | null;
  avatarUrl: string | null;
  color?: string | null;
  status?: AvatarStatus;
  size?: number;
  className?: string;
}

export function ExpertAvatar({
  name,
  avatarUrl,
  color,
  status = "idle",
  size = 40,
  className,
}: Props) {
  const style = { width: size, height: size };
  const managed = getManagedAvatar(avatarUrl, size);

  if (managed) {
    return (
      <ManagedExpertImage
        key={`${managed.base}:${size}`}
        name={name ?? "Expert"}
        base={managed.base}
        pixels={managed.pixels}
        size={size}
        isOtto={managed.assetID === "otto"}
        className={className}
      />
    );
  }

  if (!name || !avatarUrl) {
    return (
      <div
        role="img"
        aria-label={name ? `${name}, AI Expert` : "AI Expert"}
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
    return (
      <NotionAvatarImage
        config={config}
        status={status}
        size={size}
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
