import { ReactNode, useId } from "react";

interface Props {
  title: string;
  count?: number;
  description?: string;
  /** A smaller heading for detail sections that sit below the story. */
  quiet?: boolean;
  children: ReactNode;
}

export function ExpertSection({
  title,
  count,
  description,
  quiet = false,
  children,
}: Props) {
  const headingId = useId();

  return (
    <section aria-labelledby={headingId}>
      <h2
        id={headingId}
        className={
          quiet
            ? "flex items-baseline gap-2 text-lead font-semibold tracking-title text-foreground-emphasis"
            : "flex items-baseline gap-2 text-heading-md font-semibold tracking-heading text-foreground-strong"
        }
      >
        {title}
        {count !== undefined ? (
          <span className="text-body-lg font-normal text-foreground-subtle tabular-nums">
            {count}
          </span>
        ) : null}
      </h2>
      {description ? (
        <p
          className={
            quiet
              ? "mt-1 text-body leading-5 text-foreground-tertiary"
              : "mt-1.5 text-body-lg leading-6 text-foreground-tertiary"
          }
        >
          {description}
        </p>
      ) : null}
      <div className="mt-4">{children}</div>
    </section>
  );
}
