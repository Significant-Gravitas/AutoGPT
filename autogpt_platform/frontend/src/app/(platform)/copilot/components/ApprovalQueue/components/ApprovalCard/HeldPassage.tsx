interface Props {
  passage: string;
}

export function HeldPassage({ passage }: Props) {
  return (
    <figure className="flex flex-col gap-1">
      <figcaption className="text-xs text-foreground-tertiary">
        What it says
      </figcaption>
      <blockquote className="whitespace-pre-wrap border-l-2 border-warning bg-warning-surface/60 px-3 py-2 text-sm text-foreground-emphasis [overflow-wrap:anywhere]">
        {passage}
      </blockquote>
    </figure>
  );
}
