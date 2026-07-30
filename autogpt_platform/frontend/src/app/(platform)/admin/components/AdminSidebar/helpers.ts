import {
  BrainIcon,
  CalculatorIcon,
  CurrencyDollarIcon,
  FileTextIcon,
  GaugeIcon,
  HeartbeatIcon,
  MagnifyingGlassIcon,
  ReceiptIcon,
  RobotIcon,
  SlidersHorizontalIcon,
  UsersIcon,
  type Icon as PhosphorIcon,
} from "@phosphor-icons/react";

export interface AdminNavItem {
  label: string;
  href: string;
  Icon: PhosphorIcon;
}

export const adminNavItems: AdminNavItem[] = [
  {
    label: "Marketplace Management",
    href: "/admin/marketplace",
    Icon: UsersIcon,
  },
  { label: "User Spending", href: "/admin/spending", Icon: CurrencyDollarIcon },
  {
    label: "System Diagnostics",
    href: "/admin/diagnostics",
    Icon: HeartbeatIcon,
  },
  {
    label: "User Impersonation",
    href: "/admin/impersonation",
    Icon: MagnifyingGlassIcon,
  },
  { label: "Rate Limits", href: "/admin/rate-limits", Icon: GaugeIcon },
  { label: "Platform Costs", href: "/admin/platform-costs", Icon: ReceiptIcon },
  {
    label: "Execution Analytics",
    href: "/admin/execution-analytics",
    Icon: FileTextIcon,
  },
  { label: "Bot Analytics", href: "/admin/bots", Icon: RobotIcon },
  {
    label: "Block Cost Estimates",
    href: "/admin/block-cost-estimates",
    Icon: CalculatorIcon,
  },
  { label: "Memory Inspector", href: "/admin/memory", Icon: BrainIcon },
  {
    label: "Admin User Management",
    href: "/admin/settings",
    Icon: SlidersHorizontalIcon,
  },
];
