"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";

import {
  IconShoppingCart,
  IconBoxes,
  IconLibrary,
  IconLaptop,
} from "@/components/ui/icons";

interface NavbarLinkProps {
  name: string;
  href: string;
}

const icons = {
  "/marketplace": IconShoppingCart,
  "/build": IconBoxes,
  "/library": IconLibrary,
  "/home": IconLibrary,
};

export const NavbarLink = ({ name, href }: NavbarLinkProps) => {
  const pathname = usePathname();
  const isActive = pathname === href;

  const Icon = icons[href as keyof typeof icons];

  return (
    <Link href={href} data-testid={`navbar-link-${name.toLowerCase()}`}>
      <div
        className={`flex h-10 items-center justify-start gap-2 px-3 py-2 ${
          isActive ? "rounded-lg bg-zinc-800 dark:bg-neutral-200" : ""
        }`}
      >
        {Icon && (
          <Icon
            className={`h-5 w-5 ${
              isActive ? "text-zinc-50 dark:text-black" : ""
            }`}
          />
        )}
        <div
          className={`hidden font-poppins text-base font-medium lg:block ${
            isActive
              ? "text-zinc-50 dark:text-neutral-900"
              : "text-neutral-900 dark:text-neutral-50"
          }`}
        >
          {name}
        </div>
      </div>
    </Link>
  );
};
