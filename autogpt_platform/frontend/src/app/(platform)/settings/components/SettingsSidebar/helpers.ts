import {
  ChartLineUpIcon,
  CreditCardIcon,
  IdentificationBadgeIcon,
  KeyIcon,
  PlugsConnectedIcon,
  SlidersHorizontalIcon,
  UserIcon,
  type Icon as PhosphorIcon,
} from "@phosphor-icons/react";

export interface SettingsNavItem {
  label: string;
  href: string;
  Icon: PhosphorIcon;
}

export const settingsNavItems: SettingsNavItem[] = [
  { label: "Profile", href: "/settings/profile", Icon: UserIcon },
  {
    label: "Creator Dashboard",
    href: "/settings/creator-dashboard",
    Icon: ChartLineUpIcon,
  },
  { label: "Billing", href: "/settings/billing", Icon: CreditCardIcon },
  {
    label: "Integrations",
    href: "/settings/integrations",
    Icon: PlugsConnectedIcon,
  },
  {
    label: "Preferences",
    href: "/settings/preferences",
    Icon: SlidersHorizontalIcon,
  },
  { label: "AutoGPT API Keys", href: "/settings/api-keys", Icon: KeyIcon },
  {
    label: "OAuth Apps",
    href: "/settings/oauth-apps",
    Icon: IdentificationBadgeIcon,
  },
];
