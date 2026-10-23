import { cn } from "@/lib/utils";
import Link from "next/link";
import { ReactNode } from "react";

interface Props {
  media: ReactNode;
  mediaClassName?: string;
  title: string;
  subtitle: string;
  trailing?: ReactNode;
  testId?: string;
  /** A tile is one target: a link when it navigates, a button when it opens. */
  href?: string;
  onClick?: () => void;
}

const TILE =
  "flex w-full items-center gap-3 rounded-xl border border-zinc-200/80 bg-white p-2.5 text-left outline-none transition-colors hover:border-zinc-300 focus-visible:ring-2 focus-visible:ring-violet-600";

export function ShelfTile({
  media,
  mediaClassName,
  title,
  subtitle,
  trailing,
  testId,
  href,
  onClick,
}: Props) {
  const body = (
    <>
      <span
        className={cn(
          "relative flex h-14 w-14 shrink-0 items-center justify-center overflow-hidden rounded-lg",
          mediaClassName,
        )}
      >
        {media}
      </span>
      <span className="min-w-0 flex-1">
        <span
          title={title}
          className="block truncate text-sm font-medium text-zinc-900"
        >
          {title}
        </span>
        <span className="block truncate text-[13px] text-zinc-500">
          {subtitle}
        </span>
      </span>
      {trailing}
    </>
  );

  return (
    <li className="min-w-0">
      {href ? (
        <Link href={href} data-testid={testId} className={TILE}>
          {body}
        </Link>
      ) : (
        <button
          type="button"
          onClick={onClick}
          data-testid={testId}
          className={TILE}
        >
          {body}
        </button>
      )}
    </li>
  );
}
