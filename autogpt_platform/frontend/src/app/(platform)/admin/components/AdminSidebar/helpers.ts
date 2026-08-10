import {
  BrainIcon,
  Calculator01Icon,
  DollarSignIcon,
  File02Icon,
  GaugeIcon,
  Pulse01Icon,
  ReceiptTextIcon,
  Robot01Icon,
  Search01Icon,
  SlidersHorizontalIcon,
  UserMultipleIcon,
} from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";

export interface AdminNavItem {
  label: string;
  href: string;
  Icon: IconSvgElement;
}

export const adminNavItems: AdminNavItem[] = [
  {
    label: "Marketplace Management",
    href: "/admin/marketplace",
    Icon: UserMultipleIcon,
  },
  { label: "User Spending", href: "/admin/spending", Icon: DollarSignIcon },
  {
    label: "System Diagnostics",
    href: "/admin/diagnostics",
    Icon: Pulse01Icon,
  },
  {
    label: "User Impersonation",
    href: "/admin/impersonation",
    Icon: Search01Icon,
  },
  { label: "Rate Limits", href: "/admin/rate-limits", Icon: GaugeIcon },
  {
    label: "Platform Costs",
    href: "/admin/platform-costs",
    Icon: ReceiptTextIcon,
  },
  {
    label: "Execution Analytics",
    href: "/admin/execution-analytics",
    Icon: File02Icon,
  },
  { label: "Bot Analytics", href: "/admin/bots", Icon: Robot01Icon },
  {
    label: "Block Cost Estimates",
    href: "/admin/block-cost-estimates",
    Icon: Calculator01Icon,
  },
  { label: "Memory Inspector", href: "/admin/memory", Icon: BrainIcon },
  {
    label: "Admin User Management",
    href: "/admin/settings",
    Icon: SlidersHorizontalIcon,
  },
];
