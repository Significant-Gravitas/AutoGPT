import {
  ChartLineUpIcon,
  ChatsCircleIcon,
  CreditCardIcon,
  KeyIcon,
  PaintBrushIcon,
  PlugsConnectedIcon,
  SlidersHorizontalIcon,
  UserIcon,
  type Icon as PhosphorIcon,
} from "@/components/atoms/AGPTIcon/icons";

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
  {
    label: "Appearance",
    href: "/settings/appearance",
    Icon: PaintBrushIcon,
  },
  { label: "Billing", href: "/settings/billing", Icon: CreditCardIcon },
  {
    label: "Integrations",
    href: "/settings/integrations",
    Icon: PlugsConnectedIcon,
  },
  { label: "Bots", href: "/settings/bots", Icon: ChatsCircleIcon },
  { label: "AutoGPT API Keys", href: "/settings/api-keys", Icon: KeyIcon },
  {
    label: "Creator Dashboard",
    href: "/settings/creator-dashboard",
    Icon: ChartLineUpIcon,
  },
];
