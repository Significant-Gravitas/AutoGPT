import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import {
  Alert02Icon,
  CancelCircleIcon,
  CheckmarkCircle02Icon,
  InformationCircleIcon,
} from "@hugeicons/core-free-icons";

import { Icon } from "@/components/atoms/Icon/Icon";
import { cn } from "@/lib/utils";

const alertVariants = cva(
  "relative w-full rounded-lg border px-4 py-3 text-sm text-card-foreground [&>svg]:absolute [&>svg]:top-1/2 [&>svg]:left-4 [&>svg]:-translate-y-1/2 [&>svg~*]:pl-7",
  {
    variants: {
      variant: {
        default: "border-border bg-card [&>svg]:text-accent",
        info: "border-border bg-info-foreground [&>svg]:text-info",
        success: "border-success/30 bg-success/10 [&>svg]:text-success",
        warning: "border-warning/40 bg-warning/10 [&>svg]:text-warning",
        error:
          "border-destructive/30 bg-destructive/10 [&>svg]:text-destructive",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  },
);

const variantIcons = {
  default: InformationCircleIcon,
  info: InformationCircleIcon,
  success: CheckmarkCircle02Icon,
  warning: Alert02Icon,
  error: CancelCircleIcon,
} as const;

interface AlertProps
  extends
    React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof alertVariants> {
  children: React.ReactNode;
  /** Override the default variant icon (e.g. a domain-specific icon). */
  icon?: React.ComponentType<{ className?: string }>;
}

const Alert = React.forwardRef<HTMLDivElement, AlertProps>(
  (
    { className, variant = "default", icon: CustomIcon, children, ...props },
    ref,
  ) => {
    const currentVariant = variant || "default";
    const iconClassName = "h-4.5 w-4.5";

    return (
      <div
        ref={ref}
        role="alert"
        className={cn(alertVariants({ variant: currentVariant }), className)}
        {...props}
      >
        {CustomIcon ? (
          <CustomIcon className={iconClassName} />
        ) : (
          <Icon icon={variantIcons[currentVariant]} className={iconClassName} />
        )}
        {children}
      </div>
    );
  },
);
Alert.displayName = "Alert";

const AlertTitle = React.forwardRef<
  HTMLParagraphElement,
  React.HTMLAttributes<HTMLHeadingElement>
>(({ className, ...props }, ref) => (
  <h5
    ref={ref}
    className={cn("mb-1 leading-none font-medium tracking-tight", className)}
    {...props}
  />
));

AlertTitle.displayName = "AlertTitle";

const AlertDescription = React.forwardRef<
  HTMLParagraphElement,
  React.HTMLAttributes<HTMLParagraphElement>
>(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={cn("text-sm [&_p]:leading-relaxed", className)}
    {...props}
  />
));
AlertDescription.displayName = "AlertDescription";

export { Alert, AlertTitle, AlertDescription };
