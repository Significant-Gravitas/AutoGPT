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
      // TODO(tokens): no role for Tailwind's default gray-600; the zinc roles
      // are a different grey.
      className={cn("leading-4 text-gray-600", className)}
      {...rest}
    >
      {children}
    </Text>
  );
}
