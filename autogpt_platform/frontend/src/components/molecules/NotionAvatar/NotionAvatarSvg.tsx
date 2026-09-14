import { cn } from "@/lib/utils";
import { composeNotionAvatar } from "./compose";
import {
  encodeNotionConfig,
  findNotionColor,
  type NotionAvatarConfig,
} from "./helpers";
import type { AvatarStatus } from "./status";

interface Props {
  config: NotionAvatarConfig;
  status: AvatarStatus;
  idPrefix: string;
  size?: number;
  showBadge?: boolean;
  transparent?: boolean;
  title?: string;
  className?: string;
}

// The same markup the /avatars/notion route serves, inlined so a picker can
// redraw without a round trip. One composer, so the two can never disagree.
export function NotionAvatarSvg({
  config,
  status,
  idPrefix,
  size = 96,
  showBadge = true,
  transparent = false,
  title,
  className,
}: Props) {
  const color = findNotionColor(config.color);

  return (
    <span
      role="img"
      aria-label={title ?? `${color.label} avatar`}
      data-testid="notion-avatar"
      data-avatar={encodeNotionConfig(config)}
      data-status={status}
      className={cn("inline-flex shrink-0", className)}
      dangerouslySetInnerHTML={{
        __html: composeNotionAvatar(config, {
          size,
          idPrefix,
          transparent,
          status: showBadge ? status : "idle",
        }),
      }}
    />
  );
}
