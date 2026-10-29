import { cva, type VariantProps } from "class-variance-authority";

import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";

// Inlined from the former shadcn button-group primitive (SECRT-3113 How 3):
// MessageBranch* were its only consumer, so it lives here, private to Message.
const buttonGroupVariants = cva(
  "flex w-fit items-stretch has-[>[data-slot=button-group]]:gap-2 focus-visible:[&>*]:relative focus-visible:[&>*]:z-10 [&>[data-slot=select-trigger]:last-of-type]:has-[select[aria-hidden=true]:last-child]:rounded-r-md [&>[data-slot=select-trigger]:not([class*='w-'])]:w-fit [&>input]:flex-1",
  {
    variants: {
      orientation: {
        horizontal:
          "[&>*:not(:first-child)]:rounded-l-none [&>*:not(:first-child)]:border-l-0 [&>*:not(:last-child)]:rounded-r-none",
        vertical:
          "flex-col [&>*:not(:first-child)]:rounded-t-none [&>*:not(:first-child)]:border-t-0 [&>*:not(:last-child)]:rounded-b-none",
      },
    },
    defaultVariants: {
      orientation: "horizontal",
    },
  },
);

export function ButtonGroup({
  className,
  orientation,
  ...props
}: React.ComponentProps<"div"> & VariantProps<typeof buttonGroupVariants>) {
  return (
    <div
      role="group"
      data-slot="button-group"
      data-orientation={orientation}
      className={cn(buttonGroupVariants({ orientation }), className)}
      {...props}
    />
  );
}

export function ButtonGroupText({
  className,
  ...props
}: React.ComponentProps<"div">) {
  return (
    <Text
      variant="body-medium-tight"
      tone="inherit"
      as="div"
      className={cn(
        // TODO(tokens): no exact role for border-border-base #E5E5E5, bg-surface-hover
        // #F5F5F5 or dark:border-action / dark:bg-action #262626
        "shadow-xs flex items-center gap-2 rounded-md border border-border-base bg-surface-hover px-4 [&_svg:not([class*='size-'])]:size-4 [&_svg]:pointer-events-none",
        className,
      )}
      {...props}
    />
  );
}
