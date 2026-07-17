import Image from "next/image";
import Link from "next/link";
import type { ReactNode } from "react";

interface Props {
  /** Page-specific title rendered in the title slot.  Truncated on overflow. */
  title?: ReactNode;
  /** Page-specific subtitle below the title (e.g. "Shared May 12, 2026"). */
  subtitle?: ReactNode;
  /** Page-specific CTAs / pills rendered in the actions slot. */
  actions?: ReactNode;
}

// Shared header bar for every ``/share/...`` route.
//
// The same DOM reflows between breakpoints via CSS grid template
// areas — no JS breakpoint hook, no duplicated markup (so only one
// h1 lives in the document regardless of viewport).  First paint
// matches final paint, so no hydration flicker.
//
//  - Mobile  (< sm): two rows — logo spans the top, title + actions
//                    sit on the bottom row.
//  - Desktop (>= sm): one row — title left, logo centre, actions right.
export function ShareHeader({ title, subtitle, actions }: Props) {
  return (
    <header
      className={
        "grid shrink-0 grid-cols-[1fr_auto] grid-rows-[auto_auto] gap-x-4 " +
        "gap-y-2 border-b border-border bg-background px-4 py-3 " +
        "[grid-template-areas:'logo_logo'_'title_actions'] " +
        "sm:grid-cols-[1fr_auto_1fr] sm:grid-rows-1 " +
        "sm:items-center sm:[grid-template-areas:'title_logo_actions']"
      }
    >
      <div className="min-w-0 [grid-area:title]">
        {title && (
          <h1 className="truncate text-sm font-semibold text-zinc-900">
            {title}
          </h1>
        )}
        {subtitle && (
          <p className="truncate text-xs text-zinc-500">{subtitle}</p>
        )}
      </div>
      <div className="flex justify-center [grid-area:logo]">
        <Logo />
      </div>
      <div className="flex shrink-0 items-center justify-end gap-2 [grid-area:actions]">
        {actions}
      </div>
    </header>
  );
}

function Logo() {
  return (
    <Link href="/" className="inline-block">
      <Image
        src="/autogpt-logo-dark-bg.png"
        alt="AutoGPT"
        width={120}
        height={54}
        className="hidden h-7 w-auto dark:block"
      />
      <Image
        src="/autogpt-logo-light-bg.png"
        alt="AutoGPT"
        width={120}
        height={54}
        className="block h-7 w-auto dark:hidden"
        priority
      />
    </Link>
  );
}
