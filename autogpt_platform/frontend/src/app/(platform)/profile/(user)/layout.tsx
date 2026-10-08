"use client";

import * as React from "react";
import {
  SettingsNav,
  type SettingsNavLink,
} from "@/components/layout/SettingsNav/SettingsNav";
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
import { useNewSettingsRedirect } from "./useNewSettingsRedirect";

export default function Layout({ children }: { children: React.ReactNode }) {
  const isPaymentEnabled = useGetFlag(Flag.ENABLE_PLATFORM_PAYMENT);
  const { isRedirecting } = useNewSettingsRedirect();

  const links: SettingsNavLink[] = [
    { label: "Profile", href: "/profile", icon: UserCircleIcon },
    {
      label: "Creator Dashboard",
      href: "/profile/dashboard",
      icon: Store01Icon,
    },
    ...(isPaymentEnabled
      ? [{ label: "Billing", href: "/profile/credits", icon: Coins01Icon }]
      : []),
    {
      label: "Integrations",
      href: "/profile/integrations",
      icon: ElectricPlugsIcon,
    },
    {
      label: "Settings",
      href: "/profile/settings",
      icon: SlidersHorizontalIcon,
    },
    { label: "API Keys", href: "/profile/api-keys", icon: Key01Icon },
    { label: "OAuth Apps", href: "/profile/oauth-apps", icon: AppWindowIcon },
  ];

  // These legacy pages redirect to /settings — render nothing while the
  // replace() is in flight so the old shell never flashes.
  if (isRedirecting) return null;

  return (
    <div className="flex min-h-screen w-full max-w-[1360px] flex-col gap-4 lg:flex-row">
      <SettingsNav label="Settings" links={links} />
      <div className="min-w-0 flex-1">{children}</div>
    </div>
  );
}
