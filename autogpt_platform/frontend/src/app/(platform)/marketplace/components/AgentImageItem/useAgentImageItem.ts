import { useEffect, useRef } from "react";

interface UseAgentImageItemProps {
  playingVideoIndex: number | null;
  index: number;
}

export const useAgentImageItem = ({
  playingVideoIndex,
  index,
}: UseAgentImageItemProps) => {
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    if (
      playingVideoIndex !== index &&
      videoRef.current &&
      !videoRef.current.paused
    ) {
      videoRef.current.pause();
    }
  }, [playingVideoIndex, index]);

  return {
    videoRef,
  };
};
