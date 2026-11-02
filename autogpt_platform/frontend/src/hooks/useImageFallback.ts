import { useEffect, useState } from "react";

// Treats an image URL that fails to load as absent, so call sites render their
// own placeholder. Resets on src change, which a bare error flag misses.
export function useImageFallback(src: string | null | undefined) {
  const [hasFailed, setHasFailed] = useState(false);

  useEffect(
    function clearFailureOnSrcChange() {
      setHasFailed(false);
    },
    [src],
  );

  return {
    showImage: Boolean(src) && !hasFailed,
    handleImageError: () => setHasFailed(true),
  };
}
