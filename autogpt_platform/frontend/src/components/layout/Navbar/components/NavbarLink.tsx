"use client";

import { IconLaptop } from "@/components/ui/icons";
import { cn } from "@/lib/utils";
import {
  CubeIcon,
  HouseIcon,
  StorefrontIcon,
} from "@phosphor-icons/react/dist/ssr";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Text } from "../../../atoms/Text/Text";

interface Props {
  name: string;
  href: string;
}

export function NavbarLink({ name, href }: Props) {
  const pathname = usePathname();
  const parts = pathname.split("/");
  const activeLink = "/" + (parts.length > 2 ? parts[2] : parts[1]);
  const isActive = activeLink === href;

  return (
    <Link
      href={href}
      data-testid={`navbar-link-${name.toLowerCase()}`}
      className="font-poppins text-[20px] leading-[28px]"
    >
      <div
        className={cn(
          "flex items-center justify-start gap-1 p-2",
          isActive &&
            "rounded-small bg-neutral-800 p-2 transition-all duration-300 dark:bg-neutral-200",
        )}
      >
        {href === "/marketplace" && (
          <StorefrontIcon
            className={cn("h-6 w-6", isActive && "text-white dark:text-black")}
          />
        )}
        {href === "/build" && (
          <CubeIcon
            className={cn("h-6 w-6", isActive && "text-white dark:text-black")}
          />
        )}
        {href === "/monitor" && (
          <IconLaptop
            className={cn("h-6 w-6", isActive && "text-white dark:text-black")}
          />
        )}
        {href === "/library" && (
          <HouseIcon
            className={cn("h-6 w-6", isActive && "text-white dark:text-black")}
          />
        )}
        <Text
          variant="body-medium"
          className={cn(
            "hidden lg:block",
            isActive ? "!text-white" : "!text-black",
          )}
        >
          {name}
        </Text>
      </div>
    </Link>
  );
}
