"use client";

import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { cn } from "@/lib/utils";
import { useEffect, useState } from "react";
import { AgentImageItem } from "../AgentImageItem/AgentImageItem";
import { getYouTubeVideoId, isValidVideoUrl } from "../AgentImageItem/helpers";
import { useAgentImage } from "./useAgentImage";

interface AgentImagesProps {
  images: string[];
}

export function AgentImages({ images }: AgentImagesProps) {
  const { playingVideoIndex, handlePlay, handlePause } = useAgentImage();
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [loadedThumbs, setLoadedThumbs] = useState<Set<number>>(new Set());

  useEffect(() => {
    setSelectedIndex((prev) => Math.max(0, Math.min(prev, images.length - 1)));
  }, [images.length]);

  if (images.length === 0) return null;

  return (
    <div className="w-full px-2 dark:bg-transparent lg:w-3/5 lg:flex-1">
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
            const isVideo = isValidVideoUrl(image);
            const youtubeId = isVideo ? getYouTubeVideoId(image) : null;

            return (
              <button
                key={index}
                type="button"
                onClick={() => setSelectedIndex(index)}
                className={cn(
                  "relative h-16 w-24 shrink-0 overflow-hidden rounded-lg border transition-all sm:h-20 sm:w-32",
                  selectedIndex === index
                    ? "border-violet-500"
                    : "border-zinc-100 opacity-70 hover:opacity-100",
                )}
              >
                {(!isVideo || youtubeId) && !loadedThumbs.has(index) && (
                  <Skeleton className="absolute inset-0 rounded-lg" />
                )}
                {youtubeId ? (
                  <img
                    src={`https://img.youtube.com/vi/${youtubeId}/mqdefault.jpg`}
                    alt={`Thumbnail ${index + 1}`}
                    loading="lazy"
                    className="absolute inset-0 h-full w-full object-cover"
                    onLoad={() =>
                      setLoadedThumbs((prev) => new Set(prev).add(index))
                    }
                  />
                ) : isVideo ? (
                  <div className="flex h-full w-full items-center justify-center bg-neutral-200 text-xs text-neutral-500">
                    Video
                  </div>
                ) : (
                  <img
                    src={image}
                    alt={`Thumbnail ${index + 1}`}
                    loading="lazy"
                    className="absolute inset-0 h-full w-full object-cover"
                    onLoad={() =>
                      setLoadedThumbs((prev) => new Set(prev).add(index))
                    }
                  />
                )}
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}
