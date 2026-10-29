"use client";

import { useState } from "react";
import Link from "next/link";
import { ArrowDown01Icon, ArrowLeft02Icon } from "@hugeicons/core-free-icons";
import { Icon as UIIcon } from "@/components/atoms/Icon/Icon";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/molecules/Popover/Popover";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import type { SectionNavLink } from "./SectionNavItem";

interface Props {
  items: (SectionNavLink & { isActive: boolean })[];
  sectionName: string;
  showBackLink?: boolean;
}

export function SectionMobileNav({ items, sectionName, showBackLink }: Props) {
  const [open, setOpen] = useState(false);
  const current = items.find((i) => i.isActive) ?? items[0];

  return (
    <div
      className={cn(
        "bg-surface-muted px-4 py-3 md:hidden",
        showBackLink && "flex items-center gap-2",
      )}
    >
      {showBackLink ? (
        <Link
          href="/copilot"
          aria-label="Back to home"
          className="flex items-center gap-1 rounded-lg py-1 pr-2 text-foreground-body transition-colors hover:text-foreground-base"
        >
          <UIIcon icon={ArrowLeft02Icon} size={16} />
        </Link>
      ) : null}
      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger asChild>
          <button
            type="button"
            className="flex w-fit items-center gap-2 rounded-full border border-border-base bg-surface px-3 py-2 outline-none focus-visible:ring-2 focus-visible:ring-action"
            aria-label={`${sectionName} navigation, current: ${current.label}`}
          >
            <span className="flex items-center gap-2">
              <UIIcon
                icon={current.Icon}
                size={16}
                className="text-foreground-base"
              />
              <Text variant="body" as="span" className="font-medium">
                {current.label}
              </Text>
            </span>
            <UIIcon
              icon={ArrowDown01Icon}
              size={16}
              className={cn(
                "text-foreground-body transition-transform",
                open && "rotate-180",
              )}
            />
          </button>
        </PopoverTrigger>
        <PopoverContent
          align="start"
          sideOffset={8}
          className="w-[calc(100vw-32px)] max-w-sm p-2"
        >
          <nav className="flex flex-col gap-1">
            {items.map(({ label, href, Icon, isActive }) => (
              <Link
                key={href}
                href={href}
                aria-current={isActive ? "page" : undefined}
                onClick={() => setOpen(false)}
                className={cn(
                  "flex h-[38px] items-center gap-2 rounded-lg px-3",
                  isActive ? "bg-surface-hover" : "hover:bg-surface-muted",
                )}
              >
                <UIIcon
                  icon={Icon}
                  size={16}
                  className="text-foreground-base"
                />
                <Text
                  variant="body"
                  as="span"
                  className={cn(
                    "flex-1",
                    isActive
                      ? "font-medium text-foreground-base"
                      : "font-normal text-foreground-body",
                  )}
                >
                  {label}
                </Text>
              </Link>
            ))}
          </nav>
        </PopoverContent>
      </Popover>
    </div>
  );
}
