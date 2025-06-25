import { cn } from "@/lib/utils";
import NextLink from "next/link";
import { forwardRef } from "react";

interface LinkProps {
  href: string;
  children: React.ReactNode;
  className?: string;
  isExternal?: boolean;
}

const Link = forwardRef<HTMLAnchorElement, LinkProps>(function Link(
  { href, children, className, isExternal = false, ...props },
  ref,
) {
  const linkClasses = cn(
    // Base styles from Figma
    "font-['Geist'] text-sm font-medium leading-[22px] text-[var(--AutoGPT-Text-text-black,#141414)]",
    // Hover state
    "hover:underline",
    // Focus state for accessibility
    "focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 rounded-sm",
    className,
  );

  if (isExternal) {
    return (
      <a
        ref={ref}
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
    <NextLink ref={ref} href={href} className={linkClasses} {...props}>
      {children}
    </NextLink>
  );
});

export { Link };
