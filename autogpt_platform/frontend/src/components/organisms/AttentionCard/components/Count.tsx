import type { ComponentPropsWithoutRef, ReactNode } from "react";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";

interface CountProps extends ComponentPropsWithoutRef<"span"> {
  children: ReactNode;
}

export function Count({ className, children, ...rest }: CountProps) {
  return (
    <Text
      data-slot="attention-card-count"
      variant="small"
      as="span"
      className={cn("leading-4 text-foreground-secondary", className)}
      {...rest}
    >
      {children}
    </Text>
  );
}
