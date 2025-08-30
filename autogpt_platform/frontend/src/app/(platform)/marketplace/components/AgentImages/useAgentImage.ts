import { useCallback, useState } from "react";

export const useAgentImage = () => {
  const [playingVideoIndex, setPlayingVideoIndex] = useState<number | null>(
    null,
  );

  const handlePlay = useCallback((index: number) => {
    setPlayingVideoIndex(index);
  }, []);

  const handlePause = useCallback(
    (index: number) => {
      if (playingVideoIndex === index) {
        setPlayingVideoIndex(null);
      }
    },
    [playingVideoIndex],
  );

  return {
    playingVideoIndex,
    handlePlay,
    handlePause,
  };
};
