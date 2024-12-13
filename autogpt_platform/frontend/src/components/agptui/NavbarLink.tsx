"use client";
import Link from "next/link";
import {
  IconType,
  IconShoppingCart,
  IconBoxes,
  IconLibrary,
  IconLaptop,
} from "@/components/ui/icons";
import { usePathname } from "next/navigation";

interface NavbarLinkProps {
  name: string;
  href: string;
}

export const NavbarLink = ({ name, href }: NavbarLinkProps) => {
  const pathname = usePathname();
  const parts = pathname.split("/");
  const activeLink = "/" + (parts.length > 2 ? parts[2] : parts[1]);

  return (
    <div
      className={`px-5 py-4 ${
        activeLink === href
          ? "rounded-2xl bg-neutral-800 dark:bg-neutral-200"
          : ""
      } flex items-center justify-start gap-3`}
    >
      {href === "/store" && (
        <IconShoppingCart
          className={`h-6 w-6 ${activeLink === href ? "text-white dark:text-black" : ""}`}
        />
      )}
      {href === "/build" && (
        <IconBoxes
          className={`h-6 w-6 ${activeLink === href ? "text-white dark:text-black" : ""}`}
        />
      )}
      {href === "/monitor" && (
        <IconLaptop
          className={`h-6 w-6 ${activeLink === href ? "text-white dark:text-black" : ""}`}
        />
      )}
      {href === "/monitoring" && (
        <IconLibrary
          className={`h-6 w-6 ${activeLink === href ? "text-white dark:text-black" : ""}`}
        />
      )}
      <Link href={href}>
        <div
          className={`font-poppins text-xl font-medium leading-7 ${
            activeLink === href
              ? "text-neutral-50 dark:text-neutral-900"
              : "text-neutral-900 dark:text-neutral-50"
          }`}
        >
          {name}
        </div>
      </Link>
    </div>
  );
};
