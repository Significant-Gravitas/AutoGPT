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
        "min-w-0 rounded-lg bg-gradient-to-r from-brand/30 to-info/30 p-px",
        className,
      )}
    >
      <div className="rounded-lg bg-surface-hover p-3">{children}</div>
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
      <div className="min-w-0 flex-1">{children}</div>
      {action}
    </div>
  );
}

export function ContentCardTitle({
  children,
  size = "default",
  className,
}: {
  children: React.ReactNode;
  size?: "default" | "small";
  className?: string;
}) {
  return (
    <Text
      variant={size === "small" ? "small-medium-tight" : "body-medium"}
      tone="emphasis"
      className={cn("truncate", className)}
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
      mono
      tone="emphasis"
      className={cn("mt-0.5 truncate", className)}
    >
      {children}
    </Text>
  );
}

export function ContentCardDescription({
  children,
  size = "default",
  className,
}: {
  children: React.ReactNode;
  size?: "default" | "small";
  className?: string;
}) {
  return (
    <Text
      variant={size === "small" ? "small-tight" : "body"}
      tone="emphasis"
      className={cn("mt-2", className)}
    >
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
    <Text variant="body" tone="emphasis" className={className}>
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
    <Text variant="small" tone="muted" className={className}>
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
        "rounded-lg border bg-surface-hover p-3 text-caption whitespace-pre-wrap text-foreground-emphasis",
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
        "shrink-0 rounded-full border bg-surface-hover px-2 py-0.5 text-caption-sm leading-inherit text-foreground-emphasis",
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
      className={cn("shrink-0 text-caption text-brand", className)}
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
        "mt-2 list-disc space-y-1 pl-5 font-sans text-caption leading-4.5 text-foreground-emphasis",
        className,
      )}
    >
      {items.slice(0, max).map((s) => (
        <li key={s}>{s}</li>
      ))}
    </ul>
  );
}
