import { useState, useEffect } from "react";

export const useMediaQuery = (): boolean => {
  const [matches, setMatches] = useState(false);

  useEffect(() => {
    const query = "(max-width: 768px)";
    const media = window.matchMedia(query);
    if (media.matches !== matches) {
      setMatches(media.matches);
    }
    const listener = () => setMatches(media.matches);
    media.addEventListener("change", listener);
    return () => media.removeEventListener("change", listener);
  }, [matches]);

  return matches;
};
