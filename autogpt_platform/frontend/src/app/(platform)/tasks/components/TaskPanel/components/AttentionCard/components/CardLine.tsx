import { Text } from "@/components/atoms/Text/Text";
import type { ReactNode } from "react";

interface Props {
  label: string;
  children: ReactNode;
}

/** "Why: …" or "What I need: …", in full, never cut off. */
export function CardLine({ label, children }: Props) {
  return (
    <Text
      variant="inbox"
      tone="ink"
      className="break-words whitespace-pre-wrap"
    >
      <Text as="span" variant="inbox-medium" tone="ink">
        {label}:
      </Text>{" "}
      <span>{children}</span>
    </Text>
  );
}
