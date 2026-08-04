import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { cn } from "@/lib/utils";
import { ArrowSquareOutIcon } from "@phosphor-icons/react";
import Link, { useLinkStatus } from "next/link";
import * as React from "react";

interface Props {
  icon: React.ReactNode;
  label: string;
  href?: string;
  onClick?: () => void;
  destructive?: boolean;
  as?: "link" | "button";
  external?: boolean;
  // New sidebar layout variant: lighter text weight + external-link glyph.
  newLayout?: boolean;
}

const baseRowClasses =
  "group relative flex w-full items-center gap-3 rounded-lg pl-3 pr-2 py-2 text-left text-sm outline-none transition-colors duration-200 ease-out focus-visible:outline-none";

function RowBody({
  icon,
  label,
  destructive,
  external = false,
  pending = false,
}: {
  icon: React.ReactNode;
  label: string;
  destructive: boolean;
  external?: boolean;
  pending?: boolean;
}) {
  const barClasses = destructive
    ? "absolute left-0 top-1/2 -translate-y-1/2 w-[3px] h-5 rounded-full bg-red-500 opacity-0 transition-opacity duration-200 group-hover:opacity-100 group-focus-visible:opacity-100"
    : "absolute left-0 top-1/2 -translate-y-1/2 w-[3px] h-5 rounded-full bg-neutral-900 opacity-0 transition-opacity duration-200 group-hover:opacity-100 group-focus-visible:opacity-100";

  return (
    <>
      <span className={barClasses} aria-hidden="true" />
      <span className="relative z-10 flex shrink-0 items-center">{icon}</span>
      <span className="relative z-10 flex-1 truncate">{label}</span>
      {pending ? (
        <LoadingSpinner
          size="small"
          className="relative z-10 text-current"
          aria-hidden="true"
        />
      ) : external ? (
        <ArrowSquareOutIcon
          className="relative z-10 shrink-0 text-neutral-700"
          size={16}
          aria-hidden="true"
        />
      ) : null}
    </>
  );
}

function LinkRowBody({
  icon,
  label,
  destructive,
  external = false,
}: {
  icon: React.ReactNode;
  label: string;
  destructive: boolean;
  external?: boolean;
}) {
  const { pending } = useLinkStatus();
  return (
    <RowBody
      icon={icon}
      label={label}
      destructive={destructive}
      external={external}
      pending={pending}
    />
  );
}

export function AccountMenuRow({
  icon,
  label,
  href,
  onClick,
  destructive = false,
  as = "link",
  external = false,
  newLayout = false,
}: Props) {
  const colorClasses = destructive
    ? "text-neutral-700 hover:bg-red-50 hover:text-red-600 focus-visible:bg-red-50 focus-visible:text-red-600"
    : "text-neutral-700 hover:bg-neutral-100 focus-visible:bg-neutral-100";
  const rowClasses = cn(
    baseRowClasses,
    newLayout ? "font-normal" : "font-medium",
    colorClasses,
  );

  if (as === "link" && href) {
    if (external) {
      return (
        <a
          href={href}
          target="_blank"
          rel="noopener noreferrer"
          className={rowClasses}
        >
          <RowBody
            icon={icon}
            label={label}
            destructive={destructive}
            external={newLayout}
          />
        </a>
      );
    }
    return (
      <Link href={href} className={rowClasses}>
        <LinkRowBody icon={icon} label={label} destructive={destructive} />
      </Link>
    );
  }

  return (
    <button type="button" onClick={onClick} className={rowClasses}>
      <RowBody icon={icon} label={label} destructive={destructive} />
    </button>
  );
}
