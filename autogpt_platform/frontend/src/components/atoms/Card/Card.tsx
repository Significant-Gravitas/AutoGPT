import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { HTMLAttributes, ReactNode } from "react";

const variants = {
  default: "rounded-large bg-surface p-6 shadow-md",
  outlined: "agpt-card text-foreground-base",
  // A small bordered box for a list of help rows under a form.
  panel:
    "agpt-card text-foreground-base rounded-lg border border-border-base bg-surface shadow-sm",
} as const;

interface Props {
  children: ReactNode;
  className?: string;
  variant?: keyof typeof variants;
  /** `roomy`: 32px inside, for a card that is the whole page (an auth or
   *  error screen). */
  size?: "roomy";
}

export function Card({
  children,
  className,
  variant = "default",
  size,
}: Props) {
  return (
    <div
      data-slot="card"
      className={cn(variants[variant], size === "roomy" && "p-8", className)}
    >
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
      variant="body"
      leading="5"
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
  padding,
  ...props
}: HTMLAttributes<HTMLDivElement> & {
  /** `none`: rows that run to the card's edges. */
  padding?: "none";
}) {
  return (
    <div
      data-slot="card-content"
      className={cn("p-6 pt-0", padding === "none" && "p-0", className)}
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
