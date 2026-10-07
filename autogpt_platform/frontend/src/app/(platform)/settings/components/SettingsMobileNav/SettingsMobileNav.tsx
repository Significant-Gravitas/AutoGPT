"use client";

import { useState } from "react";
import Link from "next/link";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/molecules/Popover/Popover";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { useSettingsSidebar } from "../SettingsSidebar/useSettingsSidebar";
import { ArrowDown01Icon } from "@hugeicons/core-free-icons";
import { Icon as UIIcon } from "@/components/atoms/Icon/Icon";

export function SettingsMobileNav() {
  const { items } = useSettingsSidebar();
  const [open, setOpen] = useState(false);
  const current = items.find((i) => i.isActive) ?? items[0];

  return (
    <div className="bg-zinc-50 px-4 py-3 md:hidden">
      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger asChild>
          <button
            type="button"
            className="flex w-fit items-center gap-2 rounded-full border border-zinc-200 bg-white px-3 py-2 outline-hidden focus-visible:ring-2 focus-visible:ring-zinc-800"
            aria-label={`Settings navigation, current: ${current.label}`}
          >
            <span className="flex items-center gap-2">
              <UIIcon icon={current.Icon} size={16} className="text-black" />
              <Text variant="body-medium" as="span">
                {current.label}
              </Text>
            </span>
            <UIIcon
              icon={ArrowDown01Icon}
              size={16}
              className={cn(
                "text-zinc-700 transition-transform",
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
                  isActive ? "bg-zinc-100" : "hover:bg-zinc-50",
                )}
              >
                <UIIcon icon={Icon} size={16} className="text-black" />
                <Text
                  variant={isActive ? "body-medium" : "body"}
                  as="span"
                  tone={isActive ? undefined : "secondary"}
                  className="flex-1"
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
