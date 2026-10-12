import { cn } from "@/lib/utils";
import Image from "next/image";
import { notionAvatarImageUrlFor, type NotionAvatarConfig } from "./helpers";
import type { AvatarStatus } from "./status";
import { StatusDot } from "./StatusDot";

interface Props {
  config: NotionAvatarConfig;
  status?: AvatarStatus;
  size?: number;
  title?: string;
  className?: string;
}

/** The avatar as a cacheable image. The face never moves, so this is how it is
 *  drawn everywhere except the picker, which redraws as you cycle features. */
export function NotionAvatarImage({
  config,
  status = "idle",
  size = 36,
  title,
  className,
}: Props) {
  const src = notionAvatarImageUrlFor(config);
  const face = (
    <Image
      src={src}
      alt={title ?? "Avatar"}
      width={size}
      height={size}
      unoptimized
      data-testid="notion-avatar-image"
      data-avatar={src}
      className={cn("shrink-0 rounded-full", className)}
    />
  );

  if (status === "idle") return face;

  return (
    <span
      style={{ width: size, height: size }}
      className="relative inline-flex shrink-0"
    >
      {face}
      <StatusDot
        status={status}
        size={Math.round(size * 0.34)}
        className="absolute -bottom-0.5 -right-0.5"
      />
    </span>
  );
}
