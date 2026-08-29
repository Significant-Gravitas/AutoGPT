import { useBreakpoint } from "@/lib/hooks/useBreakpoint";

export function useIsMobile() {
  const breakpoint = useBreakpoint();
  return breakpoint === "base" || breakpoint === "sm" || breakpoint === "md";
}
