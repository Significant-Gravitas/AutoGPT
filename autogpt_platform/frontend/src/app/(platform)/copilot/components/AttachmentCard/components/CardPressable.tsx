import { cn } from "@/lib/utils";
import Link from "next/link";
import type { ReactNode } from "react";

interface Props {
  onOpen?: () => void;
  href?: string;
  title?: string;
  className?: string;
  children: ReactNode;
}

/** The card's body: a link when it navigates, a button when it opens the
 *  file, and plain content when it does neither. */
export function CardPressable({
  onOpen,
  href,
  title,
  className,
  children,
}: Props) {
  const pressableClassName = cn(
    "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-zinc-900",
    className,
  );
  if (href)
    return (
      <Link href={href} title={title} className={pressableClassName}>
        {children}
      </Link>
    );
  if (onOpen)
    return (
      <button
        type="button"
        onClick={onOpen}
        title={title}
        className={pressableClassName}
      >
        {children}
      </button>
    );
  return (
    <div title={title} className={className}>
      {children}
    </div>
  );
}

export function isPressable({ onOpen, href }: Pick<Props, "onOpen" | "href">) {
  return Boolean(onOpen || href);
}
