import { Text } from "@/components/atoms/Text/Text";
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
      <Text
        variant="lead-semibold"
        as="h2"
        tone="primary"
        id={headingId}
        className="flex items-baseline gap-2 tracking-[-0.02em]"
      >
        {title}
        {count !== undefined ? (
          <span className="text-base font-normal tabular-nums text-zinc-400">
            {count}
          </span>
        ) : null}
      </Text>
      {description ? (
        <Text variant="large" tone="muted" className="mt-1.5 leading-6">
          {description}
        </Text>
      ) : null}
      <div className="mt-4">{children}</div>
    </section>
  );
}
