"use client";

// Transitions.dev shimmer-text: ::before duplicates the string via
// content:attr(data-text) and sweeps a highlight band clipped to the glyphs.
interface Props {
  text: string;
  className?: string;
}

export function ShimmerText({ text, className }: Props) {
  return (
    <span
      data-text={text}
      className={
        "relative inline-block text-zinc-500 " +
        "before:pointer-events-none before:absolute before:inset-0 before:content-[attr(data-text)] " +
        "before:animate-shimmer-text before:bg-[linear-gradient(90deg,transparent_0%,transparent_40%,#18181b_50%,transparent_60%,transparent_100%)] " +
        "before:bg-[length:400%_100%] before:bg-clip-text before:bg-no-repeat before:text-transparent " +
        "motion-reduce:before:animate-none " +
        (className ?? "")
      }
    >
      {text}
    </span>
  );
}
