import { Text } from "@/components/atoms/Text/Text";
import { ReactNode } from "react";

export function SoulSectionTitle({ children }: { children: ReactNode }) {
  return (
    <Text variant="body-medium" as="h3" tone="primary">
      {children}
    </Text>
  );
}
