import { cn } from "@/lib/utils";

interface Props {
  size?: number;
  className?: string;
}

export function ScaleLoader({ size = 48, className }: Props) {
  return (
    <div
      className={cn(
        "relative inline-block size-12",
        // oxlint-disable-next-line design-system/no-arbitrary-values -- the scale-loader keyframes in globals.css
        "before:absolute before:top-0 before:left-0 before:box-border before:size-full before:animate-[scale-loader-grow_2s_linear_infinite_backwards] before:rounded-full before:bg-current before:content-['']",
        // oxlint-disable-next-line design-system/no-arbitrary-values -- the scale-loader keyframes in globals.css
        "after:absolute after:top-0 after:left-0 after:box-border after:size-full after:animate-[scale-loader-grow_2s_linear_1s_infinite_backwards] after:rounded-full after:bg-current after:content-['']",
        className,
      )}
      style={{ width: size, height: size }}
    />
  );
}
