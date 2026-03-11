"use client";
import { AgentImageItem } from "../AgentImageItem/AgentImageItem";
import { useAgentImage } from "./useAgentImage";

interface AgentImagesProps {
  images: string[];
}

export const AgentImages: React.FC<AgentImagesProps> = ({ images }) => {
  const { playingVideoIndex, handlePlay, handlePause } = useAgentImage();
  return (
    <div className="w-full overflow-y-auto bg-white px-2 lg:w-225 dark:bg-transparent">
      <div className="space-y-4 sm:space-y-6 md:space-y-7.5">
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
    </div>
  );
};
