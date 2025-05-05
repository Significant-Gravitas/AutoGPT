"use client";

import * as React from "react";
import { AgentImageItem } from "./AgentImageItem";

interface AgentImagesProps {
  images: string[];
}

export const AgentImages: React.FC<AgentImagesProps> = ({ images }) => {
  const [playingVideoIndex, setPlayingVideoIndex] = React.useState<
    number | null
  >(null);

  const handlePlay = React.useCallback((index: number) => {
    setPlayingVideoIndex(index);
  }, []);

  const handlePause = React.useCallback(
    (index: number) => {
      if (playingVideoIndex === index) {
        setPlayingVideoIndex(null);
      }
    },
    [playingVideoIndex],
  );

  return (
    <div className="w-full space-y-4">
      {images.map((image, index) => (
        <AgentImageItem
          key={index}
          image={image}
          index={index}
          playingVideoIndex={playingVideoIndex}
          handlePlay={handlePlay}
          handlePause={handlePause}
        />
      ))}
    </div>
  );
};
