import { useBreakpoint } from "@/lib/hooks/useBreakpoint";
import { Key, storage } from "@/services/storage/local-storage";
import { useEffect, useState } from "react";

export function useMobileWarning() {
  const breakpoint = useBreakpoint();
  const [isDismissed, setIsDismissed] = useState(false);
  // Default true so the warning never flashes open before the mount effect has
  // had a chance to read the localStorage flag.
  const [isSuppressed, setIsSuppressed] = useState(true);
  const [isReady, setIsReady] = useState(false);

  useEffect(() => {
    setIsSuppressed(storage.get(Key.BUILDER_MOBILE_WARNING_SUPPRESSED) === "1");
    // Defer the open-flip past the first paint so the Dialog's internal
    // breakpoint hook has settled — otherwise it briefly mounts the desktop
    // modal variant on mobile before switching to the drawer.
    const id = requestAnimationFrame(() => setIsReady(true));
    return () => cancelAnimationFrame(id);
  }, []);

  const isMobile =
    breakpoint === "base" || breakpoint === "sm" || breakpoint === "md";

  const isOpen = isReady && isMobile && !isDismissed && !isSuppressed;

  function dismiss() {
    setIsDismissed(true);
  }

  function suppress() {
    storage.set(Key.BUILDER_MOBILE_WARNING_SUPPRESSED, "1");
    setIsSuppressed(true);
  }

  return { isOpen, dismiss, suppress };
}
