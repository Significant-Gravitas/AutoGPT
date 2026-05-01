import {
  ChartLineUpIcon,
  CreditCardIcon,
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
    label: "Account",
    href: "/settings/account",
    Icon: SlidersHorizontalIcon,
  },
  { label: "Billing", href: "/settings/billing", Icon: CreditCardIcon },
  {
    label: "Integrations",
    href: "/settings/integrations",
    Icon: PlugsConnectedIcon,
  },
  { label: "AutoGPT API Keys", href: "/settings/api-keys", Icon: KeyIcon },
  {
    label: "Creator Dashboard",
    href: "/settings/creator-dashboard",
    Icon: ChartLineUpIcon,
  },
];
