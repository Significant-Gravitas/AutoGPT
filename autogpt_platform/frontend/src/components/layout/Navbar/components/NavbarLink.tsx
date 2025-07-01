"use client";
import { IconLaptop } from "@/components/ui/icons";
import {
  CubeIcon,
  HouseIcon,
  StorefrontIcon,
} from "@phosphor-icons/react/dist/ssr";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Text } from "../../../atoms/Text/Text";

interface NavbarLinkProps {
  name: string;
  href: string;
}

export const NavbarLink = ({ name, href }: NavbarLinkProps) => {
  const pathname = usePathname();
  const parts = pathname.split("/");
  const activeLink = "/" + (parts.length > 2 ? parts[2] : parts[1]);

  return (
    <Link
      href={href}
      data-testid={`navbar-link-${name.toLowerCase()}`}
      className="font-poppins text-[20px] leading-[28px]"
    >
      <div
        className={`p-2 ${
          activeLink === href
            ? "rounded-2xl bg-neutral-800 dark:bg-neutral-200"
            : ""
        } flex items-center justify-start gap-1`}
      >
        {href === "/marketplace" && (
          <StorefrontIcon
            className={`h-6 w-6 ${activeLink === href ? "text-white dark:text-black" : ""}`}
          />
        )}
        {href === "/build" && (
          <CubeIcon
            className={`h-6 w-6 ${activeLink === href ? "text-white dark:text-black" : ""}`}
          />
        )}
        {href === "/monitor" && (
          <IconLaptop
            className={`h-6 w-6 ${activeLink === href ? "text-white dark:text-black" : ""}`}
          />
        )}
        {href === "/library" && (
          <HouseIcon
            className={`h-6 w-6 ${activeLink === href ? "text-white dark:text-black" : ""}`}
          />
        )}
        <Text
          variant="body-medium"
          className={`hidden lg:block ${
            activeLink === href
              ? "text-neutral-50 dark:text-neutral-900"
              : "text-neutral-900 dark:text-neutral-50"
          }`}
        >
          {name}
        </Text>
      </div>
    </Link>
  );
};
