import { useEffect, useState } from "react";

export type Breakpoint = "base" | "sm" | "md" | "lg" | "xl" | "2xl";

// Explicitly maps to tailwind breakpoints
const breakpoints: Record<Breakpoint, number> = {
  base: 0,
  sm: 640,
  md: 768,
  lg: 1024,
  xl: 1280,
  "2xl": 1536,
};

export function useBreakpoint(): Breakpoint {
  const [breakpoint, setBreakpoint] = useState<Breakpoint>("lg");

  useEffect(() => {
    const getBreakpoint = () => {
      const width = window.innerWidth;
      if (width < breakpoints.sm) return "base";
      if (width < breakpoints.md) return "sm";
      if (width < breakpoints.lg) return "md";
      if (width < breakpoints.xl) return "lg";
      if (width < breakpoints["2xl"]) return "xl";
      return "2xl";
    };

    const handleResize = () => {
      const current = getBreakpoint();
      setBreakpoint(current);
    };

    window.addEventListener("resize", handleResize);
    handleResize(); // initial call

    return () => window.removeEventListener("resize", handleResize);
  }, []);

  return breakpoint;
}

export function isLargeScreen(bp: Breakpoint) {
  if (bp === "sm") return false;
  if (bp === "md") return false;
  if (bp === "lg") return true;
  if (bp === "xl") return true;
  if (bp === "2xl") return true;
  return false;
}
