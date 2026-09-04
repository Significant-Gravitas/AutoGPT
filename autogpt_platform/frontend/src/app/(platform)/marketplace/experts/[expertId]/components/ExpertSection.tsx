import { ReactNode } from "react";

interface Props {
  title: string;
  count?: number;
  description?: string;
  children: ReactNode;
}

export function ExpertSection({ title, count, description, children }: Props) {
  return (
    <section>
      <h2 className="flex items-baseline gap-2 text-base font-medium text-zinc-900">
        {title}
        {count !== undefined ? (
          <span className="text-sm font-normal tabular-nums text-zinc-400">
            {count}
          </span>
        ) : null}
      </h2>
      {description ? (
        <p className="mt-1 text-sm leading-5 text-zinc-500">{description}</p>
      ) : null}
      <div className="mt-3">{children}</div>
    </section>
  );
}
