"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { YoutubeIcon } from "@hugeicons/core-free-icons";
import { getNoticeText } from "./helpers";
import { useYouTubeEmbed } from "./useYouTubeEmbed";

interface Props {
  /** Any YouTube link; one that names no single video only gets a link out. */
  url: string;
  title?: string;
  className?: string;
}

// YouTube sets advertising cookies (Cookiebot's Marketing category) as soon as
// its player loads, so the iframe waits for that consent. Nothing else here
// requests anything from YouTube.
export function YouTubeEmbed({
  url,
  title = "YouTube video",
  className,
}: Props) {
  const {
    videoId,
    embedUrl,
    watchUrl,
    showPlayer,
    canLoadOnce,
    canAllowHere,
    settingsReady,
    openSettings,
    loadVideo,
  } = useYouTubeEmbed(url);

  if (showPlayer && embedUrl) {
    return (
      <iframe
        data-slot="youtube-embed"
        className={cn("h-full w-full", className)}
        src={embedUrl}
        title={title}
        allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
        allowFullScreen
        sandbox="allow-scripts allow-same-origin allow-presentation allow-popups allow-popups-to-escape-sandbox"
      />
    );
  }

  return (
    <div
      role="group"
      aria-label={title}
      data-slot="youtube-embed"
      className={cn(
        "flex h-full w-full flex-col items-center justify-center gap-3 overflow-auto bg-surface-hover p-4 text-center",
        className,
      )}
    >
      <Icon
        data-slot="youtube-embed-icon"
        icon={YoutubeIcon}
        size={28}
        className="text-foreground-subtle"
      />
      <Text
        variant="body"
        tone="secondary"
        data-slot="youtube-embed-notice"
        className="max-w-sm"
      >
        {getNoticeText({ hasVideoId: videoId !== null, canAllowHere })}
      </Text>
      <div
        data-slot="youtube-embed-actions"
        className="flex flex-wrap items-center justify-center gap-2"
      >
        {watchUrl && (
          <Button
            as="NextLink"
            href={watchUrl}
            target="_blank"
            rel="noopener noreferrer"
            variant="secondary"
            size="small"
            leadingIcon={YoutubeIcon}
            data-slot="youtube-embed-watch"
          >
            Watch on YouTube
          </Button>
        )}
        {canLoadOnce && (
          <Button
            variant="primary"
            size="small"
            onClick={loadVideo}
            data-slot="youtube-embed-load"
          >
            Load video
          </Button>
        )}
        {canAllowHere && (
          <Button
            variant="primary"
            size="small"
            disabled={!settingsReady}
            onClick={openSettings}
            data-slot="youtube-embed-settings"
          >
            Cookie settings
          </Button>
        )}
      </div>
    </div>
  );
}
