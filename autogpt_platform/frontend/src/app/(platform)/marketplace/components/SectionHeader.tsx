import Link from "next/link";
import { ReactNode } from "react";
import { ArrowRight02Icon } from "@hugeicons/core-free-icons";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";

interface Props {
  eyebrow?: string;
  eyebrowIcon?: ReactNode;
  title: string;
  titleIcon?: ReactNode;
  titleId?: string;
  subtitle?: string;
  action?: { label: string; href: string };
  /** A button above the text action, for the section's second door. */
  secondaryAction?: { label: string; href: string };
}

export function SectionHeader({
  eyebrow,
  eyebrowIcon,
  title,
  titleIcon,
  titleId,
  subtitle,
  action,
  secondaryAction,
}: Props) {
  // A phone stacks the actions under the text rather than hiding them: for
  // the skills shelf this block is the only route to authoring your own.
  return (
    <div className="mb-7 flex flex-col items-start justify-between gap-4 sm:flex-row sm:items-end">
      <div>
        {eyebrow ? (
          <div className="mb-2.5 flex items-center gap-2 text-xs font-medium uppercase tracking-[0.14em] text-violet-600">
            {eyebrowIcon}
            {eyebrow}
          </div>
        ) : null}
        <h2
          id={titleId}
          className="flex items-center gap-2.5 text-3xl font-semibold tracking-[-0.02em] text-zinc-900"
        >
          {titleIcon}
          {title}
        </h2>
        {subtitle ? (
          <p className="mt-2 text-base text-zinc-500">{subtitle}</p>
        ) : null}
      </div>
      {action || secondaryAction ? (
        <div className="flex shrink-0 flex-row items-center gap-4 sm:flex-col sm:items-end sm:gap-2">
          {secondaryAction ? (
            <Button
              as="NextLink"
              href={secondaryAction.href}
              variant="secondary"
              size="small"
            >
              {secondaryAction.label}
            </Button>
          ) : null}
          {action ? (
            <Link
              href={action.href}
              className="group flex items-center gap-1 pb-1 text-sm font-medium text-zinc-500 transition-colors hover:text-zinc-900"
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
      ) : null}
    </div>
  );
}
