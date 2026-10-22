import {
  BrainIcon,
  Calculator01Icon,
  Database01Icon,
  DollarSignIcon,
  File02Icon,
  Pulse01Icon,
  ReceiptTextIcon,
  Robot01Icon,
  SlidersHorizontalIcon,
  UserIcon,
  UserMultipleIcon,
} from "@hugeicons/core-free-icons";

import { isTestDataSurfaceEnabled } from "../../test-data/helpers";
import type { IconSvgElement } from "@hugeicons/react";
import { fixedCopy, type FixedCopy } from "@/lib/replay-privacy";

export interface AdminNavItem {
  label: FixedCopy;
  href: string;
  Icon: IconSvgElement;
}

// Built per call so the local-only Test Data link follows the live
// environment check instead of the value at module load.
export function getAdminNavItems(): AdminNavItem[] {
  return [
    { label: fixedCopy("Users"), href: "/admin/users", Icon: UserIcon },
    {
      label: fixedCopy("Marketplace Management"),
      href: "/admin/marketplace",
      Icon: UserMultipleIcon,
    },
    {
      label: fixedCopy("User Spending"),
      href: "/admin/spending",
      Icon: DollarSignIcon,
    },
    {
      label: fixedCopy("System Diagnostics"),
      href: "/admin/diagnostics",
      Icon: Pulse01Icon,
    },
    {
      label: fixedCopy("Platform Costs"),
      href: "/admin/platform-costs",
      Icon: ReceiptTextIcon,
    },
    {
      label: fixedCopy("Execution Analytics"),
      href: "/admin/execution-analytics",
      Icon: File02Icon,
    },
    {
      label: fixedCopy("Bot Analytics"),
      href: "/admin/bots",
      Icon: Robot01Icon,
    },
    {
      label: fixedCopy("Block Cost Estimates"),
      href: "/admin/block-cost-estimates",
      Icon: Calculator01Icon,
    },
    {
      label: fixedCopy("Memory Inspector"),
      href: "/admin/memory",
      Icon: BrainIcon,
    },
    {
      label: fixedCopy("Admin settings"),
      href: "/admin/settings",
      Icon: SlidersHorizontalIcon,
    },
    // Test data seeding only exists on local stacks; hide the entry point
    // everywhere else so cloud admins don't hit a guaranteed 403/404.
    ...(isTestDataSurfaceEnabled()
      ? [
          {
            label: fixedCopy("Test Data"),
            href: "/admin/test-data",
            Icon: Database01Icon,
          },
        ]
      : []),
  ];
}
