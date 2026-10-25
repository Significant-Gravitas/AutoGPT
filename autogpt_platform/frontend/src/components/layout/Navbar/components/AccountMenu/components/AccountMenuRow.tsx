import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { cn } from "@/lib/utils";
import Link, { useLinkStatus } from "next/link";
import * as React from "react";
import { LinkSquare01Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

interface Props {
  icon: React.ReactNode;
  label: string;
  href?: string;
  onClick?: () => void;
  destructive?: boolean;
  as?: "link" | "button";
  external?: boolean;
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
    ? "absolute left-0 top-1/2 -translate-y-1/2 w-[3px] h-5 rounded-full bg-error opacity-0 transition-opacity duration-200 group-hover:opacity-100 group-focus-visible:opacity-100"
    : // TODO(tokens): no exact role for bg-neutral-900 (#171717)
      "absolute left-0 top-1/2 -translate-y-1/2 w-[3px] h-5 rounded-full bg-neutral-900 opacity-0 transition-opacity duration-200 group-hover:opacity-100 group-focus-visible:opacity-100";

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
        <Icon
          icon={LinkSquare01Icon}
          // TODO(tokens): no exact role for text-neutral-700 (#404040)
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
}: Props) {
  const colorClasses = destructive
    ? // TODO(tokens): no exact role for text-neutral-700 (#404040); no fitting role for hover:text-red-600, focus-visible:text-red-600 #D93636; #D93636 is error-hover (a hover fill), not a text colour
      "text-neutral-700 hover:bg-error-surface hover:text-red-600 focus-visible:bg-error-surface focus-visible:text-red-600"
    : // TODO(tokens): no exact role for text-neutral-700 (#404040), hover:bg-neutral-100 (#F5F5F5), focus-visible:bg-neutral-100 (#F5F5F5)
      "text-neutral-700 hover:bg-neutral-100 focus-visible:bg-neutral-100";
  const rowClasses = cn(baseRowClasses, "font-normal", colorClasses);

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
            external
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
