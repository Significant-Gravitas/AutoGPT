import {
  BrainIcon,
  ChartIncreaseIcon,
  CreditCardIcon,
  Key01Icon,
  MessageMultiple01Icon,
  PlugSocketIcon,
  SlidersHorizontalIcon,
  UserIcon,
} from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";
import { Flag } from "@/services/feature-flags/use-get-flag";

export interface SettingsNavItem {
  label: string;
  href: string;
  Icon: IconSvgElement;
  flag?: Flag;
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
    flag: Flag.GRAPHITI_MEMORY,
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
    label: "Creator Dashboard",
    href: "/settings/creator-dashboard",
    Icon: ChartIncreaseIcon,
  },
];
