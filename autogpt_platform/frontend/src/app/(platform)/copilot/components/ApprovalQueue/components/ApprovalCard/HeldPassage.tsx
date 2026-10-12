interface Props {
  passage: string;
}

export function HeldPassage({ passage }: Props) {
  return (
    <figure className="flex flex-col gap-1">
      <figcaption className="text-xs text-zinc-500">What it says</figcaption>
      <blockquote className="whitespace-pre-wrap border-l-2 border-amber-400 bg-amber-50/60 px-3 py-2 text-sm text-zinc-800 [overflow-wrap:anywhere]">
        {passage}
      </blockquote>
    </figure>
  );
}
