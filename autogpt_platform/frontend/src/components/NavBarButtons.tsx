"use client";

import Link from "next/link";
import { BsBoxes } from "react-icons/bs";
import { LuLaptop } from "react-icons/lu";
import { LuShoppingCart } from "react-icons/lu";
import { cn } from "@/lib/utils";
import { usePathname } from "next/navigation";

export function NavBarButtons({ className }: { className?: string }) {
  "use client";

  const pathname = usePathname();
  const buttons = [
    {
      href: "/marketplace",
      text: "Marketplace",
      icon: <LuShoppingCart />,
    },
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
  ];

  const activeButton = buttons.find((button) => button.href === pathname);

  console.log(">>>> ", activeButton);

  return buttons.map((button) => (
    <Link
      key={button.href}
      href={button.href}
      className={cn(
        className,
        "rounded-xl p-3",
        activeButton === button ? "button bg-gray-950 text-white" : "",
      )}
    >
      {button.icon} {button.text}
    </Link>
  ));
}
