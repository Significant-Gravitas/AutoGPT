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
        "before:absolute before:left-0 before:top-0 before:box-border before:size-full before:animate-[scale-loader-grow_2s_linear_infinite_backwards] before:rounded-[50%] before:bg-current before:content-['']",
        "after:absolute after:left-0 after:top-0 after:box-border after:size-full after:animate-[scale-loader-grow_2s_linear_1s_infinite_backwards] after:rounded-[50%] after:bg-current after:content-['']",
        className,
      )}
      style={{ width: size, height: size }}
    />
  );
}
