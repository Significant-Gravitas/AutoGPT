interface Props {
  passage: string;
}

export function HeldPassage({ passage }: Props) {
  return (
    <figure className="flex flex-col gap-1">
      <figcaption className="text-xs text-muted-foreground">
        What it says
      </figcaption>
      <blockquote className="border-l-2 border-yellow-400 bg-yellow-50/60 px-3 py-2 text-sm wrap-anywhere whitespace-pre-wrap text-zinc-800">
        {passage}
      </blockquote>
    </figure>
  );
}
