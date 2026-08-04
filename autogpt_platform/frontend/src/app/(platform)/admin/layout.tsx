import { Sidebar } from "@/components/__legacy__/Sidebar";
import {
  UsersIcon,
  CurrencyDollarIcon,
  MagnifyingGlassIcon,
  GaugeIcon,
  ReceiptIcon,
  FileTextIcon,
  HeartbeatIcon,
  CalculatorIcon,
  BrainIcon,
  RobotIcon,
} from "@phosphor-icons/react/dist/ssr";

import { IconSliders } from "@/components/__legacy__/ui/icons";

const sidebarLinkGroups = [
  {
    links: [
      {
        text: "Marketplace Management",
        href: "/admin/marketplace",
        icon: <UsersIcon className="h-6 w-6" />,
      },
      {
        text: "User Spending",
        href: "/admin/spending",
        icon: <CurrencyDollarIcon className="h-6 w-6" />,
      },
      {
        text: "System Diagnostics",
        href: "/admin/diagnostics",
        icon: <HeartbeatIcon className="h-6 w-6" />,
      },
      {
        text: "User Impersonation",
        href: "/admin/impersonation",
        icon: <MagnifyingGlassIcon className="h-6 w-6" />,
      },
      {
        text: "Rate Limits",
        href: "/admin/rate-limits",
        icon: <GaugeIcon className="h-6 w-6" />,
      },
      {
        text: "Platform Costs",
        href: "/admin/platform-costs",
        icon: <ReceiptIcon className="h-6 w-6" />,
      },
      {
        text: "Execution Analytics",
        href: "/admin/execution-analytics",
        icon: <FileTextIcon className="h-6 w-6" />,
      },
      {
        text: "Bot Analytics",
        href: "/admin/bots",
        icon: <RobotIcon className="h-6 w-6" />,
      },
      {
        text: "Block Cost Estimates",
        href: "/admin/block-cost-estimates",
        icon: <CalculatorIcon className="h-6 w-6" />,
      },
      {
        text: "Memory Inspector",
        href: "/admin/memory",
        icon: <BrainIcon className="h-6 w-6" />,
      },
      {
        text: "Admin User Management",
        href: "/admin/settings",
        icon: <IconSliders className="h-6 w-6" />,
      },
    ],
  },
];

export default function AdminLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="flex h-full w-full flex-col lg:flex-row">
      <Sidebar linkGroups={sidebarLinkGroups} />
      <div className="flex-1 pl-4">{children}</div>
    </div>
  );
}
