import type { ComponentPropsWithoutRef, ReactNode } from "react";
import { Text } from "@/components/atoms/Text/Text";

interface CountProps extends ComponentPropsWithoutRef<"span"> {
  children: ReactNode;
}

export function Count({ className, children, ...rest }: CountProps) {
  return (
    <Text
      data-slot="attention-card-count"
      variant="small"
      leading="4"
      as="span"
      tone="secondary"
      className={className}
      {...rest}
    >
      {children}
    </Text>
  );
}
