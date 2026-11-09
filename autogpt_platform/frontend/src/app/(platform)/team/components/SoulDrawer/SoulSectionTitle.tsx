import { Text } from "@/components/atoms/Text/Text";
import type { ReactNode } from "react";

interface Props {
  children: ReactNode;
}

export function SoulSectionTitle({ children }: Props) {
  return (
    <Text variant="body-medium" as="h3" tone="primary">
      {children}
    </Text>
  );
}
