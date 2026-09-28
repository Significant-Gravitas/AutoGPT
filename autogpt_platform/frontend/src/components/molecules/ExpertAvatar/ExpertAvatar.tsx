import {
  Avatar,
  AvatarFallback,
  AvatarImage,
} from "@/components/atoms/Avatar/Avatar";
import type { AvatarStatus } from "@/components/molecules/NotionAvatar/status";
import { StatusDot } from "@/components/molecules/NotionAvatar/StatusDot";
import { expertPastel } from "./colors";
import { cn } from "@/lib/utils";
import { getManagedAvatar, resolveExpertAvatarUrl } from "./helpers";

import { ManagedExpertImage } from "./components/ManagedExpertImage";

interface Props {
  name: string | null;
  avatarUrl: string | null;
  color?: string | null;
  status?: AvatarStatus;
  size?: number;
  className?: string;
  /** Pastel wash behind a custom upload or generated image. A managed
   *  identity ignores it: its opaque studio tile is the artwork. */
  backgroundColor?: string;
}

/** An Expert's saved appearance. The image never depends on the name, role,
 *  category or the marketplace filter: a managed identity is served from its
 *  versioned library path, anything else is shown exactly as saved. */
export function ExpertAvatar({
  name,
  avatarUrl,
  status = "idle",
  size = 40,
  className,
  backgroundColor,
}: Props) {
  const src = resolveExpertAvatarUrl(avatarUrl);
  const managed = getManagedAvatar(src, size);
  if (managed) {
    return (
      <ManagedExpertImage
        key={`${managed.base}:${size}`}
        name={name ?? "Expert"}
        base={managed.base}
        pixels={managed.pixels}
        pngMaxPixels={managed.pngMaxPixels}
        size={size}
        isOtto={managed.assetID === "otto"}
        className={className}
      />
    );
  }
  return (
    <span
      style={{
        width: size,
        height: size,
        backgroundColor: backgroundColor
          ? expertPastel(backgroundColor)
          : undefined,
      }}
      className={cn(
        "relative inline-flex shrink-0",
        backgroundColor && "rounded-full",
        className,
      )}
    >
      <Avatar className="size-full rounded-[inherit]">
        <AvatarImage
          src={src}
          alt={name ?? "Expert"}
          width={size}
          height={size}
          className="object-contain"
        />
        <AvatarFallback
          accessibleLabel={name ? `${name}, AI Expert` : "AI Expert"}
        >
          <span className="text-sm">
            {name?.slice(0, 1).toUpperCase() ?? "?"}
          </span>
        </AvatarFallback>
      </Avatar>
      {status !== "idle" && (
        <StatusDot
          status={status}
          size={Math.round(size * 0.34)}
          className="absolute -bottom-0.5 -right-0.5"
        />
      )}
    </span>
  );
}
