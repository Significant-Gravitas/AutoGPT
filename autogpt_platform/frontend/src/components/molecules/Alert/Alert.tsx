import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import {
  Alert02Icon,
  CancelCircleIcon,
  InformationCircleIcon,
} from "@hugeicons/core-free-icons";

import { Icon } from "@/components/atoms/Icon/Icon";
import { cn } from "@/lib/utils";

const alertVariants = cva(
  "relative w-full rounded-lg border border-zinc-200 px-4 py-3 text-sm [&>svg]:absolute [&>svg]:left-4 [&>svg]:top-1/2 [&>svg]:-translate-y-1/2 [&>svg]:text-zinc-800 [&>svg~*]:pl-7",
  {
    variants: {
      variant: {
        default: "bg-white text-zinc-800 [&>svg]:text-purple-500",
        warning:
          "bg-orange-50/50 border-yellow-300 text-zinc-800 [&>svg]:text-orange-600",
        error:
          "bg-red-100/50 border-red-300 text-zinc-800 [&>svg]:text-red-500",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  },
);

const variantIcons = {
  default: InformationCircleIcon,
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
