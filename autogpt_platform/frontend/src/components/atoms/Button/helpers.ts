import {
  linkBaseClasses,
  linkFocusClasses,
  linkVariantClasses,
} from "@/components/atoms/Link/Link";
import { cn } from "@/lib/utils";
import { cva, VariantProps } from "class-variance-authority";
import { LinkProps } from "next/link";

// Extended button variants based on our design system
export const extendedButtonVariants = cva(
  "inline-flex items-center justify-center whitespace-nowrap font-medium transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-neutral-950 disabled:pointer-events-none disabled:opacity-50 font-sans leading-snug border min-w-[7.7rem]",
  {
    variants: {
      variant: {
        primary:
          "bg-zinc-800 border-zinc-800 text-white hover:bg-zinc-900 hover:border-zinc-900 rounded-full disabled:text-white disabled:bg-zinc-200 disabled:border-zinc-200 disabled:opacity-1",
        secondary:
          "bg-zinc-100 border-zinc-100 text-black hover:bg-zinc-300 hover:border-zinc-300 rounded-full disabled:text-zinc-300 disabled:bg-zinc-50 disabled:border-zinc-50 disabled:opacity-1",
        destructive:
          "bg-red-500 border-red-500 text-white hover:bg-red-600 hover:border-red-600 rounded-full disabled:text-white disabled:bg-zinc-200 disabled:border-zinc-200 disabled:opacity-1",
        outline:
          "bg-transparent border-zinc-700 text-black hover:bg-zinc-100 hover:border-zinc-700 rounded-full disabled:border-zinc-200 disabled:text-zinc-200 disabled:opacity-1",
        ghost:
          "bg-transparent border-transparent text-black hover:bg-zinc-50 hover:border-zinc-50 rounded-full disabled:text-zinc-200 disabled:opacity-1",
        icon: "bg-white text-black border border-zinc-600 hover:bg-zinc-100 rounded-[96px] disabled:opacity-1 !min-w-0",
        link: cn(
          linkBaseClasses,
          linkVariantClasses.secondary,
          linkFocusClasses,
          "inline-flex items-center gap-2 border-none bg-transparent px-0 py-0 text-left",
        ),
      },
      size: {
        small: "px-3 py-2 text-sm gap-1.5 h-[2.25rem]",
        large: "px-4 py-3 text-sm gap-2 h-[3.25rem]",
        icon: "p-3 !min-w-0",
      },
    },
    defaultVariants: {
      variant: "primary",
      size: "large",
    },
  },
);

type BaseButtonProps = {
  loading?: boolean;
  leftIcon?: React.ReactNode;
  rightIcon?: React.ReactNode;
  asChild?: boolean;
} & VariantProps<typeof extendedButtonVariants>;

type ButtonAsButton = BaseButtonProps &
  React.ButtonHTMLAttributes<HTMLButtonElement> & {
    as?: "button";
    href?: never;
  };

type ButtonAsLink = BaseButtonProps &
  Omit<React.AnchorHTMLAttributes<HTMLAnchorElement>, keyof LinkProps> &
  LinkProps & {
    as: "NextLink";
    disabled?: boolean;
  };

export type ButtonProps = ButtonAsButton | ButtonAsLink;
