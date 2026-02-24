import { Link } from "@/components/atoms/Link/Link";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";

/* ------------------------------------------------------------------ */
/*  Layout                                                            */
/* ------------------------------------------------------------------ */

export function ContentGrid({
  children,
  className,
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return <div className={cn("grid gap-2", className)}>{children}</div>;
}

/* ------------------------------------------------------------------ */
/*  Card                                                              */
/* ------------------------------------------------------------------ */

export function ContentCard({
  children,
  className,
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <div
      className={cn(
        "min-w-0 rounded-lg bg-gradient-to-r from-purple-500/30 to-blue-500/30 p-[1px]",
        className,
      )}
    >
      <div className="rounded-lg bg-neutral-100 p-3">{children}</div>
    </div>
  );
}

/** Flex row with a left content area (`children`) and an optional right‑side `action`. */
export function ContentCardHeader({
  children,
  action,
  className,
}: {
  children: React.ReactNode;
  action?: React.ReactNode;
  className?: string;
}) {
  return (
    <div className={cn("flex items-start justify-between gap-2", className)}>
      <div className="min-w-0">{children}</div>
      {action}
    </div>
  );
}

export function ContentCardTitle({
  children,
  className,
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <Text
      variant="body-medium"
      className={cn("truncate text-zinc-800", className)}
    >
      {children}
    </Text>
  );
}

export function ContentCardSubtitle({
  children,
  className,
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <Text
      variant="small"
      className={cn("mt-0.5 truncate font-mono text-zinc-800", className)}
    >
      {children}
    </Text>
  );
}

export function ContentCardDescription({
  children,
  className,
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <Text variant="body" className={cn("mt-2 text-zinc-800", className)}>
      {children}
    </Text>
  );
}

/* ------------------------------------------------------------------ */
/*  Text                                                              */
/* ------------------------------------------------------------------ */

export function ContentMessage({
  children,
  className,
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <Text variant="body" className={cn("text-zinc-800", className)}>
      {children}
    </Text>
  );
}

export function ContentHint({
  children,
  className,
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <Text variant="small" className={cn("text-neutral-500", className)}>
      {children}
    </Text>
  );
}

/* ------------------------------------------------------------------ */
/*  Code / data                                                       */
/* ------------------------------------------------------------------ */

export function ContentCodeBlock({
  children,
  className,
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <pre
      className={cn(
        "whitespace-pre-wrap rounded-lg border bg-black p-3 text-xs text-neutral-200",
        className,
      )}
    >
      {children}
    </pre>
  );
}

/* ------------------------------------------------------------------ */
/*  Inline elements                                                   */
/* ------------------------------------------------------------------ */

export function ContentBadge({
  children,
  className,
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <Text
      variant="small"
      as="span"
      className={cn(
        "shrink-0 rounded-full border bg-muted px-2 py-0.5 text-[11px] text-zinc-800",
        className,
      )}
    >
      {children}
    </Text>
  );
}

export function ContentLink({
  href,
  children,
  className,
  ...rest
}: Omit<React.ComponentProps<typeof Link>, "className"> & {
  className?: string;
}) {
  return (
    <Link
      variant="primary"
      isExternal
      href={href}
      className={cn("shrink-0 text-xs text-purple-500", className)}
      {...rest}
    >
      {children}
    </Link>
  );
}

/* ------------------------------------------------------------------ */
/*  Lists                                                             */
/* ------------------------------------------------------------------ */

export function ContentSuggestionsList({
  items,
  max = 5,
  className,
}: {
  items: string[];
  max?: number;
  className?: string;
}) {
  if (items.length === 0) return null;
  return (
    <ul
      className={cn(
        "mt-2 list-disc space-y-1 pl-5 font-sans text-[0.75rem] leading-[1.125rem] text-zinc-800",
        className,
      )}
    >
      {items.slice(0, max).map((s) => (
        <li key={s}>{s}</li>
      ))}
    </ul>
  );
}
