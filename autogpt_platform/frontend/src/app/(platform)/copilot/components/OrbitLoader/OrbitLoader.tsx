import { cn } from "@/lib/utils";

interface Props {
  size?: number;
  className?: string;
}

export function OrbitLoader({ size = 24, className }: Props) {
  const ballSize = Math.round(size * 0.4);
  const spacing = Math.round(size * 0.6);
  const gap = Math.round(size * 0.2);

  return (
    <div
      className={cn(
        // oxlint-disable-next-line design-system/no-arbitrary-values -- the orbit keyframes in globals.css
        "relative animate-[orbit-loader-rotate_1s_infinite]",
        "before:block before:h-[40%] before:w-[40%] before:rounded-full before:content-['']",
        "after:block after:h-[40%] after:w-[40%] after:rounded-full after:content-['']",
        // oxlint-disable-next-line design-system/no-arbitrary-values -- the ball geometry comes from the size prop through CSS variables
        "before:mb-[calc(var(--gap))] before:animate-[orbit-loader-ball1_1s_infinite] before:bg-foreground-subtle before:[box-shadow:calc(var(--spacing))_0_0_rgb(var(--color-primary))]",
        // oxlint-disable-next-line design-system/no-arbitrary-values -- the ball geometry comes from the size prop through CSS variables
        "after:animate-[orbit-loader-ball2_1s_infinite] after:bg-primary after:[box-shadow:calc(var(--spacing))_0_0_rgb(var(--color-foreground-subtle))]",
        className,
      )}
      style={
        {
          width: size,
          height: size,
          "--ball-size": `${ballSize}px`,
          "--spacing": `${spacing}px`,
          "--gap": `${gap}px`,
        } as React.CSSProperties
      }
    />
  );
}
