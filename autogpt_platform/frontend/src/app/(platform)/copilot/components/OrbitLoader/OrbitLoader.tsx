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
      // TODO(tokens): no exact role for #A1A1AA (Tailwind's default zinc-400;
      // the app palette's zinc-400 is #ADADB3). No fitting role for the box
      // shadows' #18181B; #18181B is primary (a fill), used for the ball, and
      // the shadows keep the hex of the keyframes moved unchanged.
      className={cn(
        "relative animate-[orbit-loader-rotate_1s_infinite]",
        "before:block before:h-[40%] before:w-[40%] before:rounded-[50%] before:content-['']",
        "after:block after:h-[40%] after:w-[40%] after:rounded-[50%] after:content-['']",
        // TODO(tokens): no exact role for before:bg-[#a1a1aa]; no fitting role for before:[box-shadow:calc(var(--spacing))_0_0_#18181b] #18181B; #18181B is primary (a fill) and secondary-foreground (a text colour)
        "before:mb-[calc(var(--gap))] before:animate-[orbit-loader-ball1_1s_infinite] before:bg-[#a1a1aa] before:[box-shadow:calc(var(--spacing))_0_0_#18181b]",
        // TODO(tokens): no exact role for after:[box-shadow:calc(var(--spacing))_0_0_#a1a1aa]
        "after:animate-[orbit-loader-ball2_1s_infinite] after:bg-primary after:[box-shadow:calc(var(--spacing))_0_0_#a1a1aa]",
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
