"use client";
import Link from "next/link";
import {
  IconType,
  IconMarketplace,
  IconBuilder,
  IconLibrary,
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
        activeLink === href ? "rounded-2xl bg-neutral-800" : ""
      } flex items-center justify-start gap-3`}
    >
      {href === "/store" && (
        <IconMarketplace
          className={`h-6 w-6 ${activeLink === href ? "text-white" : ""}`}
        />
      )}
      {href === "/build" && (
        <IconBuilder
          className={`h-6 w-6 ${activeLink === href ? "text-white" : ""}`}
        />
      )}
      {href === "/library" && (
        <IconLibrary
          className={`h-6 w-6 ${activeLink === href ? "text-white" : ""}`}
        />
      )}
      <Link href={href}>
        <div
          className={`font-['Poppins'] text-xl font-medium leading-7 ${
            activeLink === href ? "text-neutral-50" : "text-neutral-900"
          }`}
        >
          {name}
        </div>
      </Link>
    </div>
  );
};
