interface Props {
  passage: string;
}

export function HeldPassage({ passage }: Props) {
  return (
    <figure className="flex flex-col gap-1">
      <figcaption className="text-xs text-foreground-tertiary">
        What it says
      </figcaption>
      <blockquote className="border-l-2 border-warning bg-warning-surface/60 px-3 py-2 text-sm [overflow-wrap:anywhere] whitespace-pre-wrap text-foreground-emphasis">
        {passage}
      </blockquote>
    </figure>
  );
}
