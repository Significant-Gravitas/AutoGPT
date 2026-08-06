"use client";

import * as React from "react";
import { Sidebar } from "@/components/__legacy__/Sidebar";
import { useGetFlag, Flag } from "@/services/feature-flags/use-get-flag";
import {
  AppWindowIcon,
  Coins01Icon,
  ElectricPlugsIcon,
  Key01Icon,
  SlidersHorizontalIcon,
  Store01Icon,
  UserCircleIcon,
} from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

export default function Layout({ children }: { children: React.ReactNode }) {
  const isPaymentEnabled = useGetFlag(Flag.ENABLE_PLATFORM_PAYMENT);

  const sidebarLinkGroups = [
    {
      links: [
        {
          text: "Profile",
          href: "/profile",
          icon: <Icon icon={UserCircleIcon} className="size-5" />,
        },
        {
          text: "Creator Dashboard",
          href: "/profile/dashboard",
          icon: <Icon icon={Store01Icon} className="size-5" />,
        },
        ...(isPaymentEnabled
          ? [
              {
                text: "Billing",
                href: "/profile/credits",
                icon: <Icon icon={Coins01Icon} className="size-5" />,
              },
            ]
          : []),
        {
          text: "Integrations",
          href: "/profile/integrations",
          icon: <Icon icon={ElectricPlugsIcon} className="size-5" />,
        },
        {
          text: "Settings",
          href: "/profile/settings",
          icon: <Icon icon={SlidersHorizontalIcon} className="size-5" />,
        },
        {
          text: "API Keys",
          href: "/profile/api-keys",
          icon: <Icon icon={Key01Icon} className="size-5" />,
        },
        {
          text: "OAuth Apps",
          href: "/profile/oauth-apps",
          icon: <Icon icon={AppWindowIcon} className="size-5" />,
        },
      ],
    },
  ];

  return (
    <div className="flex min-h-screen w-full max-w-[1360px] flex-col lg:flex-row">
      <Sidebar linkGroups={sidebarLinkGroups} />
      <div className="flex-1 pl-4">{children}</div>
    </div>
  );
}
