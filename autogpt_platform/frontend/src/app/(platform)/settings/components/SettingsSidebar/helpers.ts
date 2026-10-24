import {
  BrainIcon,
  ChartIncreaseIcon,
  CreditCardIcon,
  Key01Icon,
  MessageMultiple01Icon,
  PlugSocketIcon,
  ShieldKeyIcon,
  SlidersHorizontalIcon,
  UserIcon,
} from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";

export interface SettingsNavItem {
  label: string;
  href: string;
  Icon: IconSvgElement;
}

export const settingsNavItems: SettingsNavItem[] = [
  { label: "Profile", href: "/settings/profile", Icon: UserIcon },
  {
    label: "Account",
    href: "/settings/account",
    Icon: SlidersHorizontalIcon,
  },
  {
    label: "Memory",
    href: "/settings/memory",
    Icon: BrainIcon,
  },
  { label: "Billing", href: "/settings/billing", Icon: CreditCardIcon },
  {
    label: "Integrations",
    href: "/settings/integrations",
    Icon: PlugSocketIcon,
  },
  { label: "Bots", href: "/settings/bots", Icon: MessageMultiple01Icon },
  { label: "AutoGPT API Keys", href: "/settings/api-keys", Icon: Key01Icon },
  {
    label: "Keyless Access",
    href: "/settings/keyless-access",
    Icon: ShieldKeyIcon,
  },
  {
    label: "Creator Dashboard",
    href: "/settings/creator-dashboard",
    Icon: ChartIncreaseIcon,
  },
];
