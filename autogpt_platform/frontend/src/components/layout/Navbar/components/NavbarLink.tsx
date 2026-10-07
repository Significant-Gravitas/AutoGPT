"use client";

import { cn } from "@/lib/utils";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Text } from "@/components/atoms/Text/Text";
import {
  BuilderIcon,
  HomepageIcon,
  MarketplaceIcon,
} from "./MenuIcon/MenuIcon";
import { CheckListIcon, LaptopIcon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

const iconBaseClass = "h-4 w-4 shrink-0";
const iconNudgedClass = "relative bottom-0.5 h-4 w-4 shrink-0";

interface Props {
  name: string;
  href: string;
}

export function NavbarLink({ name, href }: Props) {
  const pathname = usePathname();

  const isActive =
    href === "/home"
      ? pathname === "/" || pathname.startsWith("/home")
      : pathname.includes(href);

  return (
    <Link href={href} data-testid={`navbar-link-${name.toLowerCase()}`}>
      <div
        className={cn(
          "flex items-center justify-start gap-2.5 p-1 md:p-2",
          isActive &&
            "rounded-small bg-zinc-800 py-1 pl-1 pr-1.5 transition-all duration-300 md:py-[0.7rem] md:pl-2 md:pr-3",
        )}
      >
        {href === "/marketplace" && (
          <div className={cn(iconNudgedClass, isActive && "text-white")}>
            <MarketplaceIcon />
          </div>
        )}
        {href === "/build" && (
          <div className={cn(iconNudgedClass, isActive && "text-white")}>
            <BuilderIcon />
          </div>
        )}
        {href === "/monitor" && (
          <Icon
            icon={LaptopIcon}
            className={cn(iconBaseClass, isActive && "text-white")}
          />
        )}
        {href === "/home" && (
          <div className={cn(iconNudgedClass, isActive && "text-white")}>
            <HomepageIcon />
          </div>
        )}
        {href === "/library" && (
          <Icon
            icon={CheckListIcon}
            className={cn("h-5 w-5 shrink-0", isActive && "text-white")}
          />
        )}
        <Text
          variant="h5"
          className={cn(
            "hidden leading-none xl:block",
            isActive && "text-white",
          )}
        >
          {name}
        </Text>
      </div>
    </Link>
  );
}
