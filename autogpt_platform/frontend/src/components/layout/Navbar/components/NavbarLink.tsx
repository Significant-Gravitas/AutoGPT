"use client";

import { cn } from "@/lib/utils";
import { Laptop, ListChecksIcon } from "@phosphor-icons/react/dist/ssr";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Text } from "../../../atoms/Text/Text";
import {
  BuilderIcon,
  HomepageIcon,
  MarketplaceIcon,
} from "./MenuIcon/MenuIcon";

const iconBaseClass = "h-4 w-4 shrink-0";
const iconNudgedClass = "relative bottom-[2px] h-4 w-4 shrink-0";

interface Props {
  name: string;
  href: string;
}

export function NavbarLink({ name, href }: Props) {
  const pathname = usePathname();

  const isActive =
    href === "/copilot"
      ? pathname === "/" || pathname.startsWith("/copilot")
      : pathname.includes(href);

  return (
    <Link href={href} data-testid={`navbar-link-${name.toLowerCase()}`}>
      <div
        className={cn(
          "flex items-center justify-start gap-2.5 p-1 md:p-2",
          isActive &&
            "rounded-small bg-neutral-800 py-1 pl-1 pr-1.5 transition-all duration-300 md:py-[0.7rem] md:pl-2 md:pr-3",
        )}
      >
        {href === "/marketplace" && (
          <div
            className={cn(
              iconNudgedClass,
              isActive && "text-white dark:text-black",
            )}
          >
            <MarketplaceIcon />
          </div>
        )}
        {href === "/build" && (
          <div
            className={cn(
              iconNudgedClass,
              isActive && "text-white dark:text-black",
            )}
          >
            <BuilderIcon />
          </div>
        )}
        {href === "/monitor" && (
          <Laptop
            className={cn(
              iconBaseClass,
              isActive && "text-white dark:text-black",
            )}
          />
        )}
        {href === "/copilot" && (
          <div
            className={cn(
              iconNudgedClass,
              isActive && "text-white dark:text-black",
            )}
          >
            <HomepageIcon />
          </div>
        )}
        {href === "/library" && (
          <ListChecksIcon
            className={cn(
              "h-5 w-5 shrink-0",
              isActive && "text-white dark:text-black",
            )}
          />
        )}
        <Text
          variant="h5"
          className={cn(
            "hidden !font-poppins leading-none xl:block",
            isActive ? "!text-white" : "!text-black",
          )}
        >
          {name}
        </Text>
      </div>
    </Link>
  );
}
