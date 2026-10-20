"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { YoutubeIcon } from "@hugeicons/core-free-icons";
import { getYouTubeEmbedUrl, getYouTubeWatchUrl } from "./helpers";
import { useYouTubeEmbed } from "./useYouTubeEmbed";

interface Props {
  videoId: string;
  title?: string;
  className?: string;
}

// YouTube sets advertising cookies (Cookiebot's Marketing category) as soon as
// its player loads, so the iframe waits for that consent.
export function YouTubeEmbed({
  videoId,
  title = "YouTube video",
  className,
}: Props) {
  const {
    showPlayer,
    hasConsentBanner,
    canAllowHere,
    settingsReady,
    openSettings,
    loadVideo,
  } = useYouTubeEmbed(videoId);

  if (showPlayer) {
    return (
      <iframe
        className={cn("h-full w-full", className)}
        src={getYouTubeEmbedUrl(videoId)}
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
      className={cn(
        "flex h-full w-full flex-col items-center justify-center gap-3 overflow-auto bg-zinc-100 p-4 text-center",
        className,
      )}
    >
      <Icon icon={YoutubeIcon} size={28} className="text-zinc-400" />
      <Text variant="body" tone="secondary" className="max-w-sm">
        {canAllowHere
          ? "YouTube sets advertising cookies when this video loads. To watch it here, allow marketing cookies."
          : "YouTube sets advertising cookies when this video loads."}
      </Text>
      <div className="flex flex-wrap items-center justify-center gap-2">
        <Button
          as="NextLink"
          href={getYouTubeWatchUrl(videoId)}
          target="_blank"
          rel="noopener noreferrer"
          variant="secondary"
          size="small"
          leadingIcon={YoutubeIcon}
        >
          Watch on YouTube
        </Button>
        {!hasConsentBanner && (
          <Button variant="primary" size="small" onClick={loadVideo}>
            Load video
          </Button>
        )}
        {canAllowHere && (
          <Button
            variant="primary"
            size="small"
            disabled={!settingsReady}
            onClick={openSettings}
          >
            Cookie settings
          </Button>
        )}
      </div>
    </div>
  );
}
