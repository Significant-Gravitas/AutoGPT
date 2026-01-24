"use client";

import { IconLaptop } from "@/components/__legacy__/ui/icons";
import { getHomepageRoute } from "@/lib/constants";
import { cn } from "@/lib/utils";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { ListChecksIcon } from "@phosphor-icons/react/dist/ssr";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Text } from "../../../atoms/Text/Text";
import {
  BuilderIcon,
  HomepageIcon,
  MarketplaceIcon,
} from "./MenuIcon/MenuIcon";

const iconWidthClass = "h-5 w-5";

interface Props {
  name: string;
  href: string;
}

export function NavbarLink({ name, href }: Props) {
  const pathname = usePathname();
  const isChatEnabled = useGetFlag(Flag.CHAT);
  const homepageRoute = getHomepageRoute(isChatEnabled);

  const isActive =
    href === homepageRoute
      ? pathname === "/" || pathname.startsWith(homepageRoute)
      : pathname.includes(href);

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
          <div
            className={cn(
              iconWidthClass,
              isActive && "text-white dark:text-black",
            )}
          >
            <MarketplaceIcon />
          </div>
        )}
        {href === "/build" && (
          <div
            className={cn(
              iconWidthClass,
              isActive && "text-white dark:text-black",
            )}
          >
            <BuilderIcon />
          </div>
        )}
        {href === "/monitor" && (
          <IconLaptop
            className={cn(
              iconWidthClass,
              isActive && "text-white dark:text-black",
            )}
          />
        )}
        {href === "/copilot" && (
          <div
            className={cn(
              iconWidthClass,
              isActive && "text-white dark:text-black",
            )}
          >
            <HomepageIcon />
          </div>
        )}
        {href === "/library" &&
          (isChatEnabled ? (
            <ListChecksIcon
              className={cn(
                iconWidthClass,
                isActive && "text-white dark:text-black",
              )}
            />
          ) : (
            <div
              className={cn(
                iconWidthClass,
                isActive && "text-white dark:text-black",
              )}
            >
              <HomepageIcon />
            </div>
          ))}
        <Text
          variant="h5"
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
