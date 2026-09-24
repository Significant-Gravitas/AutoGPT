import {
  Avatar,
  AvatarFallback,
  AvatarImage,
} from "@/components/atoms/Avatar/Avatar";
import type { AvatarStatus } from "@/components/molecules/NotionAvatar/status";
import { StatusDot } from "@/components/molecules/NotionAvatar/StatusDot";
import { cn } from "@/lib/utils";
import { resolveExpertAvatarUrl } from "./helpers";

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
  status = "idle",
  size = 40,
  className,
}: Props) {
  const src = resolveExpertAvatarUrl(avatarUrl);
  return (
    <span
      style={{ width: size, height: size }}
      className={cn("relative inline-flex shrink-0", className)}
    >
      <Avatar className="size-full rounded-none">
        <AvatarImage
          src={src}
          alt={name ?? "Expert"}
          width={size}
          height={size}
          className="object-contain"
        />
        <AvatarFallback>
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
