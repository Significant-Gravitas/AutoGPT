import { ReactNode, useId } from "react";

interface Props {
  title: string;
  count?: number;
  description?: string;
  children: ReactNode;
}

export function ExpertSection({ title, count, description, children }: Props) {
  const headingId = useId();

  return (
    <section aria-labelledby={headingId}>
      <h2
        id={headingId}
        className="flex items-baseline gap-2 text-xl font-semibold tracking-[-0.02em] text-zinc-900"
      >
        {title}
        {count !== undefined ? (
          <span className="text-base font-normal tabular-nums text-zinc-400">
            {count}
          </span>
        ) : null}
      </h2>
      {description ? (
        <p className="mt-1.5 text-base leading-6 text-zinc-500">
          {description}
        </p>
      ) : null}
      <div className="mt-4">{children}</div>
    </section>
  );
}
