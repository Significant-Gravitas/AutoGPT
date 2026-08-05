import Link from "next/link";
import { ReactNode } from "react";
import { ArrowRight02Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

interface Props {
  eyebrow?: string;
  eyebrowIcon?: ReactNode;
  title: string;
  titleIcon?: ReactNode;
  subtitle?: string;
  action?: { label: string; href: string };
}

export function SectionHeader({
  eyebrow,
  eyebrowIcon,
  title,
  titleIcon,
  subtitle,
  action,
}: Props) {
  return (
    <div className="mb-7 flex items-end justify-between gap-4">
      <div>
        {eyebrow ? (
          <div className="mb-2.5 flex items-center gap-2 text-xs font-medium uppercase tracking-[0.14em] text-violet-600">
            {eyebrowIcon}
            {eyebrow}
          </div>
        ) : null}
        <h2 className="flex items-center gap-2.5 text-3xl font-semibold tracking-[-0.02em] text-zinc-900">
          {titleIcon}
          {title}
        </h2>
        {subtitle ? (
          <p className="mt-2 text-base text-zinc-500">{subtitle}</p>
        ) : null}
      </div>
      {action ? (
        <Link
          href={action.href}
          className="group hidden shrink-0 items-center gap-1 pb-1 text-sm font-medium text-zinc-500 transition-colors hover:text-zinc-900 sm:flex"
        >
          {action.label}
          <Icon
            icon={ArrowRight02Icon}
            size={14}
            className="transition-transform duration-200 group-hover:translate-x-0.5"
          />
        </Link>
      ) : null}
    </div>
  );
}
