"use client";

interface Props {
  text: string;
}

export function ExpertAboutSection({ text }: Props) {
  return (
    <section>
      <p className="whitespace-pre-line text-base leading-relaxed text-zinc-600">
        {text}
      </p>
    </section>
  );
}
