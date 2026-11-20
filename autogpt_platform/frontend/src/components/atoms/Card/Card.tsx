import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { HTMLAttributes, ReactNode } from "react";

const variants = {
  default: "rounded-large bg-surface p-6 shadow-md",
  outlined: "agpt-card text-foreground-base",
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
      className={cn("leading-none font-semibold tracking-tight", className)}
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
      tone="muted"
      className={className}
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
