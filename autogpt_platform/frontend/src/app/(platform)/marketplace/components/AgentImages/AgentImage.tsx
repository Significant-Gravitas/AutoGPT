"use client";

import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Icon } from "@/components/atoms/Icon/Icon";
import { ImageNotFound01Icon, YoutubeIcon } from "@hugeicons/core-free-icons";
import { cn } from "@/lib/utils";
import { useEffect, useState } from "react";
import { AgentImageItem } from "../AgentImageItem/AgentImageItem";
import { useConsent } from "@/services/consent/useConsent";
import { getThumbnail } from "./helpers";
import { useAgentImage } from "./useAgentImage";

interface AgentImagesProps {
  images: string[];
}

export function AgentImages({ images }: AgentImagesProps) {
  const { playingVideoIndex, handlePlay, handlePause } = useAgentImage();
  // YouTube thumbnails come from YouTube, so they wait for the same consent
  // as the player.
  const { advertising } = useConsent();
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [loadedThumbs, setLoadedThumbs] = useState<Set<number>>(new Set());
  // Keyed by URL: the component stays mounted when a listing's images change.
  const [failedThumbs, setFailedThumbs] = useState<Set<string>>(new Set());

  function markThumbFailed(url: string) {
    setFailedThumbs((prev) => new Set(prev).add(url));
  }

  useEffect(() => {
    setSelectedIndex((prev) => Math.max(0, Math.min(prev, images.length - 1)));
  }, [images.length]);

  if (images.length === 0) return null;

  return (
    <div className="w-full px-2 lg:w-3/5 lg:flex-1">
      {/* Main preview */}
      <AgentImageItem
        image={images[selectedIndex]}
        index={selectedIndex}
        playingVideoIndex={playingVideoIndex}
        handlePlay={handlePlay}
        handlePause={handlePause}
      />

      {/* Thumbnails */}
      {images.length > 1 && (
        <div className="mt-3 flex gap-2 overflow-x-auto pb-1 sm:mt-4 sm:gap-3">
          {images.map((image, index) => {
            const thumbnail = getThumbnail(image, advertising);
            const thumbnailFailed =
              thumbnail.kind === "image" && failedThumbs.has(thumbnail.src);

            return (
              <button
                key={index}
                type="button"
                onClick={() => setSelectedIndex(index)}
                className={cn(
                  "relative h-16 w-24 shrink-0 overflow-hidden rounded-lg border transition-all sm:h-20 sm:w-32",
                  selectedIndex === index
                    ? // TODO(tokens): no exact role: border-violet-500 #8B5CF6
                      "border-violet-500"
                    : "border-separator opacity-70 hover:opacity-100",
                )}
              >
                {thumbnail.kind === "image" &&
                  !loadedThumbs.has(index) &&
                  !thumbnailFailed && (
                    <Skeleton className="absolute inset-0 rounded-lg" />
                  )}
                {thumbnailFailed ? (
                  <div className="flex h-full w-full items-center justify-center bg-surface-hover">
                    <Icon
                      icon={ImageNotFound01Icon}
                      className="h-5 w-5 text-foreground-subtle"
                    />
                  </div>
                ) : thumbnail.kind === "image" ? (
                  <img
                    src={thumbnail.src}
                    alt={`Thumbnail ${index + 1}`}
                    loading="lazy"
                    className="absolute inset-0 h-full w-full object-cover"
                    onLoad={() =>
                      setLoadedThumbs((prev) => new Set(prev).add(index))
                    }
                    onError={() => markThumbFailed(thumbnail.src)}
                  />
                ) : thumbnail.kind === "youtube" ? (
                  <div className="flex h-full w-full items-center justify-center bg-surface-hover">
                    <Icon
                      icon={YoutubeIcon}
                      aria-hidden
                      className="h-6 w-6 text-foreground-subtle"
                    />
                    <span className="sr-only">{thumbnail.label}</span>
                  </div>
                ) : (
                  <div
                    // TODO(tokens): no exact role: bg-neutral-200 #E5E5E5, text-neutral-500 #737373
                    className="flex h-full w-full items-center justify-center bg-neutral-200 text-xs text-neutral-500"
                  >
                    Video
                  </div>
                )}
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}
