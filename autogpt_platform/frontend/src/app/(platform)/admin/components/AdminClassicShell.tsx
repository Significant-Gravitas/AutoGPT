import {
  SettingsNav,
  type SettingsNavLink,
} from "@/components/layout/SettingsNav/SettingsNav";
import {
  BrainIcon,
  Calculator01Icon,
  Database01Icon,
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
import { ReactNode } from "react";

import { isTestDataSurfaceEnabled } from "../test-data/helpers";

// Built per render so the local-only Test Data link follows the live
// environment check instead of the value at module load.
function getLinks(): SettingsNavLink[] {
  return [
    {
      label: "Marketplace Management",
      href: "/admin/marketplace",
      icon: UserMultipleIcon,
    },
    { label: "User Spending", href: "/admin/spending", icon: DollarSignIcon },
    {
      label: "System Diagnostics",
      href: "/admin/diagnostics",
      icon: Pulse01Icon,
    },
    {
      label: "User Impersonation",
      href: "/admin/impersonation",
      icon: Search01Icon,
    },
    { label: "Rate Limits", href: "/admin/rate-limits", icon: GaugeIcon },
    {
      label: "Platform Costs",
      href: "/admin/platform-costs",
      icon: ReceiptTextIcon,
    },
    {
      label: "Execution Analytics",
      href: "/admin/execution-analytics",
      icon: File02Icon,
    },
    { label: "Bot Analytics", href: "/admin/bots", icon: Robot01Icon },
    {
      label: "Block Cost Estimates",
      href: "/admin/block-cost-estimates",
      icon: Calculator01Icon,
    },
    { label: "Memory Inspector", href: "/admin/memory", icon: BrainIcon },
    {
      label: "Admin User Management",
      href: "/admin/settings",
      icon: SlidersHorizontalIcon,
    },
    // Test data seeding only exists on local stacks; hide the entry point
    // everywhere else so cloud admins don't hit a guaranteed 403/404.
    ...(isTestDataSurfaceEnabled()
      ? [
          {
            label: "Test Data",
            href: "/admin/test-data",
            icon: Database01Icon,
          },
        ]
      : []),
  ];
}

export function AdminClassicShell({ children }: { children: ReactNode }) {
  return (
    <div className="flex h-full w-full flex-col gap-4 lg:flex-row">
      <SettingsNav label="Admin" links={getLinks()} />
      <div className="min-w-0 flex-1">{children}</div>
    </div>
  );
}
