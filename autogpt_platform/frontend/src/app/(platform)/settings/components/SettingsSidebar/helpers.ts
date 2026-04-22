import {
  CreditCardIcon,
  GearIcon,
  IdentificationBadgeIcon,
  KeyIcon,
  PlugsConnectedIcon,
  SquaresFourIcon,
  UserCircleIcon,
  type Icon as PhosphorIcon,
} from "@phosphor-icons/react";

export interface SettingsNavItem {
  label: string;
  href: string;
  Icon: PhosphorIcon;
}

export const settingsNavItems: SettingsNavItem[] = [
  { label: "Profile", href: "/settings/profile", Icon: UserCircleIcon },
  {
    label: "Creator Dashboard",
    href: "/settings/creator-dashboard",
    Icon: SquaresFourIcon,
  },
  { label: "Billing", href: "/settings/billing", Icon: CreditCardIcon },
  {
    label: "Integrations",
    href: "/settings/integrations",
    Icon: PlugsConnectedIcon,
  },
  { label: "Settings", href: "/settings/preferences", Icon: GearIcon },
  { label: "AutoGPT API Keys", href: "/settings/api-keys", Icon: KeyIcon },
  {
    label: "OAuth Apps",
    href: "/settings/oauth-apps",
    Icon: IdentificationBadgeIcon,
  },
];
