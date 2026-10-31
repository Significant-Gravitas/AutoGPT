"use client";

import { extendedButtonVariants } from "@/components/atoms/Button/helpers";
import { Icon } from "@/components/atoms/Icon/Icon";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/molecules/DropdownMenu/DropdownMenu";
import {
  ArrowDown01Icon,
  UserGroupIcon,
  MagicWand01Icon,
} from "@hugeicons/core-free-icons";
import NextLink from "next/link";
import { cn } from "@/lib/utils";

export function TeamHeaderActions() {
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <button
          type="button"
          className={cn(
            extendedButtonVariants({ variant: "primary", size: "small" }),
            "min-w-0 gap-1.5 pl-4 pr-3",
          )}
        >
          Create
          <Icon icon={ArrowDown01Icon} size={16} />
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-48">
        <DropdownMenuItem asChild className="cursor-pointer">
          <NextLink
            href="/marketplace#experts"
            className="flex items-center gap-2"
          >
            <Icon icon={UserGroupIcon} size={16} className="text-neutral-500" />
            Hire an expert
          </NextLink>
        </DropdownMenuItem>
        <DropdownMenuItem asChild className="cursor-pointer">
          <NextLink href="/build" className="flex items-center gap-2">
            <Icon
              icon={MagicWand01Icon}
              size={16}
              className="text-neutral-500"
            />
            Build an agent
          </NextLink>
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
