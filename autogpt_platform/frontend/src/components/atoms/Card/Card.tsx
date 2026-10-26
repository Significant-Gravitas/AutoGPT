import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { HTMLAttributes, ReactNode } from "react";

const variants = {
  default: "rounded-large bg-surface p-6 shadow-md",
  // TODO(tokens): text-neutral-950 #0A0A0A is the `focus` role, a ring
  // colour, not a text colour (SECRT-3154).
  outlined: "agpt-card text-neutral-950",
} as const;

interface Props {
  children: ReactNode;
  className?: string;
  variant?: keyof typeof variants;
}

export function Card({ children, className, variant = "default" }: Props) {
  return (
    <div data-slot="card" className={cn(variants[variant], className)}>
      {children}
    </div>
  );
}

export function CardHeader({
  className,
  ...props
}: HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      data-slot="card-header"
      className={cn("flex flex-col space-y-1.5 p-6", className)}
      {...props}
    />
  );
}

export function CardTitle({
  className,
  ...props
}: HTMLAttributes<HTMLHeadingElement>) {
  return (
    <h3
      data-slot="card-title"
      className={cn("font-semibold leading-none tracking-tight", className)}
      {...props}
    />
  );
}

// Not <Text>: the description inherits its font family and weight from the
// card, which Text would set explicitly.
export function CardDescription({
  className,
  ...props
}: HTMLAttributes<HTMLParagraphElement>) {
  return (
    <Text
      variant="body-tight"
      as="p"
      data-slot="card-description"
      // TODO(tokens): no exact role: text-neutral-500 #737373. The dark: class
      // is the legacy card's, kept verbatim.
      className={cn("text-neutral-500 dark:text-neutral-400", className)}
      {...props}
    />
  );
}

export function CardContent({
  className,
  ...props
}: HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      data-slot="card-content"
      className={cn("p-6 pt-0", className)}
      {...props}
    />
  );
}

export function CardFooter({
  className,
  ...props
}: HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      data-slot="card-footer"
      className={cn("flex items-center p-6 pt-0", className)}
      {...props}
    />
  );
}
