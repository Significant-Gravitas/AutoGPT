"use client";

import dynamic from "next/dynamic";
import { useId } from "react";
import { findNotionColor, type NotionAvatarConfig } from "./helpers";
import type { AvatarStatus } from "./status";

// The artwork is a few hundred kilobytes of paths. Loading it with the route
// would put it on every page that shows an avatar, so it arrives as its own
// chunk on mount. Until it lands, the disc stands in.
const NotionAvatarSvg = dynamic(
  () => import("./NotionAvatarSvg").then((module) => module.NotionAvatarSvg),
  { ssr: false, loading: () => null },
);

interface Props {
  config: NotionAvatarConfig;
  status?: AvatarStatus;
  size?: number;
  showBadge?: boolean;
  transparent?: boolean;
  title?: string;
  className?: string;
}

/** A face drawn inline, for the picker, where cycling a feature has to redraw
 *  without a round trip. Everywhere else wants NotionAvatarImage. */
export function NotionAvatar({
  config,
  status = "idle",
  size = 96,
  showBadge = true,
  transparent = false,
  title,
  className,
}: Props) {
  const idPrefix = useId().replace(/:/g, "");

  return (
    <span
      style={{
        width: size,
        height: size,
        backgroundColor: transparent
          ? undefined
          : findNotionColor(config.color).disc,
      }}
      className="inline-flex shrink-0 items-center justify-center overflow-hidden rounded-full"
    >
      <NotionAvatarSvg
        config={config}
        status={status}
        idPrefix={idPrefix}
        size={size}
        showBadge={showBadge}
        transparent
        title={title}
        className={className}
      />
    </span>
  );
}
