"use client";

import React from "react";
import Link from "next/link";
import { BsBoxes } from "react-icons/bs";
import { LuLaptop, LuShoppingCart } from "react-icons/lu";
import { BehaveAs, cn } from "@/lib/utils";
import { usePathname } from "next/navigation";
import { getBehaveAs } from "@/lib/utils";
import { IconMarketplace } from "@/components/ui/icons";
import MarketPopup from "./MarketPopup";

export function NavBarButtons({ className }: { className?: string }) {
  const pathname = usePathname();
  const buttons = [
    {
      href: "/",
      text: "Monitor",
      icon: <LuLaptop />,
    },
    {
      href: "/build",
      text: "Build",
      icon: <BsBoxes />,
    },
    {
      href: "/store",
      text: "Marketplace",
      icon: <IconMarketplace />,
    },
  ];

  const isCloud = getBehaveAs() === BehaveAs.CLOUD;

  return (
    <>
      {buttons.map((button) => {
        const isActive = button.href === pathname;
        return (
          <Link
            key={button.href}
            href={button.href}
            data-testid={`${button.text.toLowerCase()}-nav-link`}
            className={cn(
              className,
              "flex items-center gap-2 rounded-xl p-3",
              isActive
                ? "bg-gray-950 text-white"
                : "text-muted-foreground hover:text-foreground",
            )}
          >
            {button.icon} {button.text}
          </Link>
        );
      })}
      {isCloud ? (
        <Link
          href="/marketplace"
          data-testid="marketplace-nav-link"
          className={cn(
            className,
            "flex items-center gap-2 rounded-xl p-3",
            pathname === "/marketplace"
              ? "bg-gray-950 text-white"
              : "text-muted-foreground hover:text-foreground",
          )}
        >
          <LuShoppingCart /> Marketplace
        </Link>
      ) : (
        <MarketPopup
          data-testid="marketplace-nav-link"
          className={cn(
            className,
            "flex items-center gap-2 rounded-xl p-3 text-muted-foreground hover:text-foreground",
          )}
        >
          <LuShoppingCart /> Marketplace
        </MarketPopup>
      )}
    </>
  );
}
