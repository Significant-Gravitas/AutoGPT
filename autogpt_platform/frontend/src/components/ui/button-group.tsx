import { cva, type VariantProps } from "class-variance-authority";

import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";

const buttonGroupVariants = cva(
  "flex w-fit items-stretch has-[>[data-slot=button-group]]:gap-2 [&>*]:focus-visible:relative [&>*]:focus-visible:z-10 has-[select[aria-hidden=true]:last-child]:[&>[data-slot=select-trigger]:last-of-type]:rounded-r-md [&>[data-slot=select-trigger]:not([class*='w-'])]:w-fit [&>input]:flex-1",
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

function ButtonGroup({
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

function ButtonGroupText({ className, ...props }: React.ComponentProps<"div">) {
  return (
    <Text
      variant="body-medium-tight"
      tone="inherit"
      as="div"
      className={cn(
        // TODO(tokens): no exact role for border-neutral-200 #E5E5E5, bg-neutral-100
        // #F5F5F5 or dark:border-neutral-800 / dark:bg-neutral-800 #262626
        "shadow-xs flex items-center gap-2 rounded-md border border-neutral-200 bg-neutral-100 px-4 dark:border-neutral-800 dark:bg-neutral-800 [&_svg:not([class*='size-'])]:size-4 [&_svg]:pointer-events-none",
        className,
      )}
      {...props}
    />
  );
}

export { ButtonGroup, ButtonGroupText, buttonGroupVariants };
