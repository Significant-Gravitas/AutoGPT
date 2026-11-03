import {
  Avatar,
  AvatarFallback,
  AvatarImage,
} from "@/components/atoms/Avatar/Avatar";
import type { AvatarStatus } from "@/components/molecules/NotionAvatar/status";
import { StatusDot } from "@/components/molecules/NotionAvatar/StatusDot";
import { expertPastel } from "./colors";
import { cn } from "@/lib/utils";
import type { CSSProperties } from "react";
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
      data-slot="expert-avatar"
      style={
        {
          "--avatar-size": `${size}px`,
          "--avatar-bg": backgroundColor
            ? expertPastel(backgroundColor)
            : undefined,
        } as CSSProperties
      }
      className={cn(
        "relative inline-flex shrink-0",
        backgroundColor && "rounded-full",
        className,
        "size-(--avatar-size)",
        backgroundColor && "bg-(--avatar-bg)",
      )}
    >
      <Avatar
        data-slot="expert-avatar-frame"
        // oxlint-disable-next-line design-system/no-arbitrary-values -- follows the wrapper's radius; Tailwind has no rounded-inherit
        className="size-full rounded-[inherit]"
      >
        <AvatarImage
          src={src}
          alt={name ?? "Expert"}
          width={size}
          height={size}
          data-slot="expert-avatar-image"
          className="object-contain"
        />
        <AvatarFallback
          data-slot="expert-avatar-fallback"
          accessibleLabel={name ? `${name}, AI Expert` : "AI Expert"}
        >
          <span
            data-slot="expert-avatar-initial"
            className="text-body leading-5"
          >
            {name?.slice(0, 1).toUpperCase() ?? "?"}
          </span>
        </AvatarFallback>
      </Avatar>
      {status !== "idle" && (
        <StatusDot
          status={status}
          size={Math.round(size * 0.34)}
          className="absolute -right-0.5 -bottom-0.5"
        />
      )}
    </span>
  );
}
