import { Text } from "@/components/atoms/Text/Text";
import type { ReactNode } from "react";

interface Props {
  title?: string | null;
  children: ReactNode;
}

export function BlockCard({ title, children }: Props) {
  return (
    <section className="min-w-0 rounded-2xl bg-gradient-to-b from-surface to-surface-muted p-5 smooth-shadow-ring-sm">
      {title ? (
        <Text
          variant="small-medium"
          tone="secondary"
          transform="uppercase"
          tracking="caps"
          className="mb-4 break-words"
        >
          {title}
        </Text>
      ) : null}
      {children}
    </section>
  );
}
