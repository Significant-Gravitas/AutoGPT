import { cn } from "@/lib/utils";
import NextLink from "next/link";
import { forwardRef } from "react";

interface LinkProps {
  href: string;
  children: React.ReactNode;
  className?: string;
  isExternal?: boolean;
  onClick?: React.MouseEventHandler<HTMLAnchorElement>;
  variant?: "primary" | "secondary";
}

export const linkBaseClasses =
  "font-sans text-body font-medium text-foreground-base";

export const linkVariantClasses: Record<
  Required<LinkProps>["variant"],
  string
> = {
  primary: "hover:underline",
  secondary: "underline",
};

export const linkFocusClasses =
  "focus:outline-none focus:ring-2 focus:ring-info focus:ring-offset-2 rounded-sm";

const Link = forwardRef<HTMLAnchorElement, LinkProps>(function Link(
  {
    href,
    children,
    className,
    isExternal = false,
    variant = "primary",
    ...props
  },
  ref,
) {
  const linkClasses = cn(
    linkBaseClasses,
    linkVariantClasses[variant],
    linkFocusClasses,
    className,
  );

  if (isExternal) {
    return (
      <a
        ref={ref}
        data-slot="link"
        href={href}
        target="_blank"
        rel="noopener noreferrer"
        className={linkClasses}
        {...props}
      >
        {children}
      </a>
    );
  }

  return (
    <NextLink
      ref={ref}
      data-slot="link"
      href={href}
      className={linkClasses}
      {...props}
    >
      {children}
    </NextLink>
  );
});

export { Link };
