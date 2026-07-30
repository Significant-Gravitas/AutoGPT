"use client";

import * as React from "react";
import { Sidebar } from "@/components/__legacy__/Sidebar";
import {
  AppWindowIcon,
  CoinsIcon,
  KeyIcon,
  PlugsIcon,
  SlidersHorizontalIcon,
  StorefrontIcon,
  UserCircleIcon,
} from "@phosphor-icons/react";
import { useGetFlag, Flag } from "@/services/feature-flags/use-get-flag";
import { useNewSettingsRedirect } from "./useNewSettingsRedirect";

export default function Layout({ children }: { children: React.ReactNode }) {
  const isPaymentEnabled = useGetFlag(Flag.ENABLE_PLATFORM_PAYMENT);
  const { isRedirecting } = useNewSettingsRedirect();

  const sidebarLinkGroups = [
    {
      links: [
        {
          text: "Profile",
          href: "/profile",
          icon: <UserCircleIcon className="size-5" />,
        },
        {
          text: "Creator Dashboard",
          href: "/profile/dashboard",
          icon: <StorefrontIcon className="size-5" />,
        },
        ...(isPaymentEnabled
          ? [
              {
                text: "Billing",
                href: "/profile/credits",
                icon: <CoinsIcon className="size-5" />,
              },
            ]
          : []),
        {
          text: "Integrations",
          href: "/profile/integrations",
          icon: <PlugsIcon className="size-5" />,
        },
        {
          text: "Settings",
          href: "/profile/settings",
          icon: <SlidersHorizontalIcon className="size-5" />,
        },
        {
          text: "API Keys",
          href: "/profile/api-keys",
          icon: <KeyIcon className="size-5" />,
        },
        {
          text: "OAuth Apps",
          href: "/profile/oauth-apps",
          icon: <AppWindowIcon className="size-5" />,
        },
      ],
    },
  ];

  // Under the new layout these legacy pages redirect to /settings — render
  // nothing while the replace() is in flight so the old shell never flashes.
  if (isRedirecting) return null;

  return (
    <div className="flex min-h-screen w-full max-w-[1360px] flex-col lg:flex-row">
      <Sidebar linkGroups={sidebarLinkGroups} />
      <div className="flex-1 pl-4">{children}</div>
    </div>
  );
}
