import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { InfoIcon, AlertTriangleIcon, XCircleIcon } from "lucide-react";

import { cn } from "@/lib/utils";

const alertVariants = cva(
  "relative w-full rounded-lg border border-border-base px-4 py-3 text-body leading-5 [&>svg]:absolute [&>svg]:left-4 [&>svg]:top-1/2 [&>svg]:-translate-y-1/2 [&>svg]:text-foreground-emphasis [&>svg~*]:pl-7",
  {
    variants: {
      variant: {
        default: "bg-surface text-foreground-emphasis [&>svg]:text-brand",
        // TODO(tokens): no role for orange-50 at 0x80 (#FFF3E680), yellow-300
        // (#F8D554) or orange-600 (#E77B00).
        warning:
          "bg-[#FFF3E680] border-yellow-300 text-foreground-emphasis [&>svg]:text-orange-600",
        // TODO(tokens): no role for red-100 at 0x80 (#FDECEC80) or red-300
        // (#F69999).
        error:
          "bg-[#FDECEC80] border-red-300 text-foreground-emphasis [&>svg]:text-error",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  },
);

const variantIcons = {
  default: InfoIcon,
  warning: AlertTriangleIcon,
  error: XCircleIcon,
} as const;

interface AlertProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof alertVariants> {
  children: React.ReactNode;
  /** Override the default variant icon (e.g. a domain-specific icon). */
  icon?: React.ComponentType<{ className?: string }>;
}

const Alert = React.forwardRef<HTMLDivElement, AlertProps>(
  ({ className, variant = "default", icon, children, ...props }, ref) => {
    const currentVariant = variant || "default";
    const IconComponent = icon ?? variantIcons[currentVariant];

    return (
      <div
        ref={ref}
        role="alert"
        data-slot="alert"
        className={cn(alertVariants({ variant: currentVariant }), className)}
        {...props}
      >
        <IconComponent
          data-slot="alert-icon"
          className="h-[1.125rem] w-[1.125rem]"
        />
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
  // TODO(tokens): no tracking role for tracking-tight (-0.025em).
  <h5
    ref={ref}
    data-slot="alert-title"
    className={cn("mb-1 font-medium leading-none tracking-tight", className)}
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
    data-slot="alert-description"
    className={cn("text-body leading-5 [&_p]:leading-relaxed", className)}
    {...props}
  />
));
AlertDescription.displayName = "AlertDescription";

export { Alert, AlertTitle, AlertDescription };
