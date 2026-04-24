"use client";

import { useState } from "react";
import Link from "next/link";
import { CaretDownIcon } from "@phosphor-icons/react";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/molecules/Popover/Popover";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { useSettingsSidebar } from "../SettingsSidebar/useSettingsSidebar";

export function SettingsMobileNav() {
  const { items } = useSettingsSidebar();
  const [open, setOpen] = useState(false);
  const current = items.find((i) => i.isActive) ?? items[0];

  return (
    <div className="bg-[#F9F9FA] px-4 py-3 md:hidden">
      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger asChild>
          <button
            type="button"
            className="flex w-fit items-center gap-2 rounded-full border border-[#DADADC] bg-white px-3 py-2 outline-none focus-visible:ring-2 focus-visible:ring-[#3E3E43]"
            aria-label={`Settings navigation, current: ${current.label}`}
          >
            <span className="flex items-center gap-2">
              <current.Icon size={16} weight="regular" className="text-black" />
              <Text
                variant="body"
                as="span"
                className="font-medium text-[#1F1F20]"
              >
                {current.label}
              </Text>
            </span>
            <CaretDownIcon
              size={16}
              weight="regular"
              className={cn(
                "text-[#505057] transition-transform",
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
          <nav className="flex flex-col gap-[4px]">
            {items.map(({ label, href, Icon, isActive }) => (
              <Link
                key={href}
                href={href}
                aria-current={isActive ? "page" : undefined}
                onClick={() => setOpen(false)}
                className={cn(
                  "flex h-[38px] items-center gap-2 rounded-[8px] px-3",
                  isActive ? "bg-[#EFEFF0]" : "hover:bg-[#F5F5F6]",
                )}
              >
                <Icon
                  size={16}
                  weight={isActive ? "regular" : "light"}
                  className={isActive ? "text-black" : "text-[#1F1F20]"}
                />
                <Text
                  variant="body"
                  as="span"
                  className={cn(
                    "flex-1",
                    isActive
                      ? "font-medium text-[#1F1F20]"
                      : "font-normal text-[#505057]",
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
