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
import { fixedCopy, type FixedCopy } from "@/lib/replay-privacy";

export interface SettingsNavItem {
  label: FixedCopy;
  href: string;
  Icon: IconSvgElement;
}

export const settingsNavItems: SettingsNavItem[] = [
  { label: fixedCopy("Profile"), href: "/settings/profile", Icon: UserIcon },
  {
    label: fixedCopy("Account"),
    href: "/settings/account",
    Icon: SlidersHorizontalIcon,
  },
  {
    label: fixedCopy("Memory"),
    href: "/settings/memory",
    Icon: BrainIcon,
  },
  {
    label: fixedCopy("Billing"),
    href: "/settings/billing",
    Icon: CreditCardIcon,
  },
  {
    label: fixedCopy("Integrations"),
    href: "/settings/integrations",
    Icon: PlugSocketIcon,
  },
  {
    label: fixedCopy("Bots"),
    href: "/settings/bots",
    Icon: MessageMultiple01Icon,
  },
  {
    label: fixedCopy("AutoGPT API Keys"),
    href: "/settings/api-keys",
    Icon: Key01Icon,
  },
  {
    label: fixedCopy("Creator Dashboard"),
    href: "/settings/creator-dashboard",
    Icon: ChartIncreaseIcon,
  },
];
