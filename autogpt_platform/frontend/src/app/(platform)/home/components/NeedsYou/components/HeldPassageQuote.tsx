interface Props {
  passage: string;
}

export function HeldPassageQuote({ passage }: Props) {
  return (
    <blockquote
      aria-label="What it says"
      className="mt-1 line-clamp-2 border-l-2 border-amber-400 pl-2 text-sm text-zinc-600 [overflow-wrap:anywhere]"
    >
      {passage}
    </blockquote>
  );
}
