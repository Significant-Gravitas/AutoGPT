"use client";

import { IconLaptop } from "@/components/__legacy__/ui/icons";
import { cn } from "@/lib/utils";
import {
  ChatsIcon,
  CubeIcon,
  HouseIcon,
  StorefrontIcon,
} from "@phosphor-icons/react/dist/ssr";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Text } from "../../../atoms/Text/Text";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";

const iconWidthClass = "h-5 w-5";

interface Props {
  name: string;
  href: string;
}

export function NavbarLink({ name, href }: Props) {
  const pathname = usePathname();
  const isActive = pathname.includes(href);
  const chat_enabled = useGetFlag(Flag.CHAT);

  return (
    <Link href={href} data-testid={`navbar-link-${name.toLowerCase()}`}>
      <div
        className={cn(
          "flex items-center justify-start gap-1 p-1 md:p-2",
          isActive &&
            "rounded-small bg-neutral-800 py-1 pl-1 pr-1.5 transition-all duration-300 dark:bg-neutral-200 md:py-2 md:pl-2 md:pr-3",
        )}
      >
        {href === "/marketplace" && (
          <StorefrontIcon
            className={cn(
              iconWidthClass,
              isActive && "text-white dark:text-black",
            )}
          />
        )}
        {href === "/build" && (
          <CubeIcon
            className={cn(
              iconWidthClass,
              isActive && "text-white dark:text-black",
            )}
          />
        )}
        {href === "/monitor" && (
          <IconLaptop
            className={cn(
              iconWidthClass,
              isActive && "text-white dark:text-black",
            )}
          />
        )}
        {href === "/library" && (
          <HouseIcon
            className={cn(
              iconWidthClass,
              isActive && "text-white dark:text-black",
            )}
          />
        )}
        {chat_enabled && href === "/chat" && (
          <ChatsIcon
            className={cn(
              iconWidthClass,
              isActive && "text-white dark:text-black",
            )}
          />
        )}
        <Text
          variant="h4"
          className={cn(
            "hidden !font-poppins lg:block",
            isActive ? "!text-white" : "!text-black",
          )}
        >
          {name}
        </Text>
      </div>
    </Link>
  );
}
