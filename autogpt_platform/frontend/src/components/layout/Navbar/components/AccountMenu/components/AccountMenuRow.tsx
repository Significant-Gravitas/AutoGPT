import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { cn } from "@/lib/utils";
import Link, { useLinkStatus } from "next/link";
import * as React from "react";

interface Props {
  icon: React.ReactNode;
  label: string;
  href?: string;
  onClick?: () => void;
  destructive?: boolean;
  as?: "link" | "button" | "div";
}

const baseRowClasses =
  "group relative flex w-full items-center gap-3 rounded-lg pl-3 pr-2 py-2 text-left text-sm font-medium outline-none transition-colors duration-200 ease-out focus-visible:outline-none";

function RowBody({
  icon,
  label,
  destructive,
  pending = false,
}: {
  icon: React.ReactNode;
  label: string;
  destructive: boolean;
  pending?: boolean;
}) {
  const barClasses = destructive
    ? "absolute left-0 top-1/2 -translate-y-1/2 w-[3px] h-5 rounded-full bg-red-500 opacity-0 transition-opacity duration-200 group-hover:opacity-100 group-focus-visible:opacity-100"
    : "absolute left-0 top-1/2 -translate-y-1/2 w-[3px] h-5 rounded-full bg-violet-500 opacity-0 transition-opacity duration-200 group-hover:opacity-100 group-focus-visible:opacity-100";

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
      ) : null}
    </>
  );
}

function LinkRowBody({
  icon,
  label,
  destructive,
}: {
  icon: React.ReactNode;
  label: string;
  destructive: boolean;
}) {
  const { pending } = useLinkStatus();
  return (
    <RowBody
      icon={icon}
      label={label}
      destructive={destructive}
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
}: Props) {
  const colorClasses = destructive
    ? "text-neutral-500 hover:bg-red-50 hover:text-red-600 focus-visible:bg-red-50 focus-visible:text-red-600"
    : "text-neutral-500 hover:bg-violet-50 hover:text-violet-700 focus-visible:bg-violet-50 focus-visible:text-violet-700";

  if (as === "link" && href) {
    return (
      <Link href={href} className={cn(baseRowClasses, colorClasses)}>
        <LinkRowBody icon={icon} label={label} destructive={destructive} />
      </Link>
    );
  }

  if (as === "button") {
    return (
      <button
        type="button"
        onClick={onClick}
        className={cn(baseRowClasses, colorClasses)}
      >
        <RowBody icon={icon} label={label} destructive={destructive} />
      </button>
    );
  }

  return (
    <div
      role="button"
      tabIndex={0}
      onClick={onClick}
      className={cn(baseRowClasses, colorClasses, "cursor-pointer")}
    >
      <RowBody icon={icon} label={label} destructive={destructive} />
    </div>
  );
}
