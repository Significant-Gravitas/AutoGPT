import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";
import { InfoIcon, AlertTriangleIcon, XCircleIcon } from "lucide-react";

import { cn } from "@/lib/utils";

const alertVariants = cva(
  "relative w-full rounded-lg border border-neutral-200 px-4 py-3 text-sm [&>svg]:absolute [&>svg]:left-4 [&>svg]:top-[12px] [&>svg]:text-neutral-950 [&>svg~*]:pl-7",
  {
    variants: {
      variant: {
        default: "bg-white text-neutral-950 [&>svg]:text-blue-500",
        warning:
          "bg-white border-orange-500/50 text-orange-600 [&>svg]:text-orange-500",
        error: "bg-white border-red-500/50 text-red-500 [&>svg]:text-red-500",
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
}

const Alert = React.forwardRef<HTMLDivElement, AlertProps>(
  ({ className, variant = "default", children, ...props }, ref) => {
    const currentVariant = variant || "default";
    const IconComponent = variantIcons[currentVariant];

    return (
      <div
        ref={ref}
        role="alert"
        className={cn(alertVariants({ variant: currentVariant }), className)}
        {...props}
      >
        <IconComponent className="h-4 w-4" />
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
    className={cn("text-sm [&_p]:leading-relaxed", className)}
    {...props}
  />
));
AlertDescription.displayName = "AlertDescription";

export { Alert, AlertTitle, AlertDescription };
