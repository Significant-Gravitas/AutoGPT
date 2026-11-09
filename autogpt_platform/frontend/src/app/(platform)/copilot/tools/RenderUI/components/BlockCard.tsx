import { Text } from "@/components/atoms/Text/Text";
import type { ReactNode } from "react";

interface Props {
  title?: string | null;
  children: ReactNode;
}

export function BlockCard({ title, children }: Props) {
  return (
    <section className="min-w-0 rounded-2xl border border-separator bg-surface p-4">
      {title ? (
        <Text variant="body-medium" tone="primary" className="mb-3">
          {title}
        </Text>
      ) : null}
      {children}
    </section>
  );
}
