import { Sidebar } from "@/components/__legacy__/Sidebar";
import { IconSliders } from "@/components/__legacy__/ui/icons";
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
  UserMultipleIcon,
} from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

const sidebarLinkGroups = [
  {
    links: [
      {
        text: "Marketplace Management",
        href: "/admin/marketplace",
        icon: <Icon icon={UserMultipleIcon} className="h-6 w-6" />,
      },
      {
        text: "User Spending",
        href: "/admin/spending",
        icon: <Icon icon={DollarSignIcon} className="h-6 w-6" />,
      },
      {
        text: "System Diagnostics",
        href: "/admin/diagnostics",
        icon: <Icon icon={Pulse01Icon} className="h-6 w-6" />,
      },
      {
        text: "User Impersonation",
        href: "/admin/impersonation",
        icon: <Icon icon={Search01Icon} className="h-6 w-6" />,
      },
      {
        text: "Rate Limits",
        href: "/admin/rate-limits",
        icon: <Icon icon={GaugeIcon} className="h-6 w-6" />,
      },
      {
        text: "Platform Costs",
        href: "/admin/platform-costs",
        icon: <Icon icon={ReceiptTextIcon} className="h-6 w-6" />,
      },
      {
        text: "Execution Analytics",
        href: "/admin/execution-analytics",
        icon: <Icon icon={File02Icon} className="h-6 w-6" />,
      },
      {
        text: "Bot Analytics",
        href: "/admin/bots",
        icon: <Icon icon={Robot01Icon} className="h-6 w-6" />,
      },
      {
        text: "Block Cost Estimates",
        href: "/admin/block-cost-estimates",
        icon: <Icon icon={Calculator01Icon} className="h-6 w-6" />,
      },
      {
        text: "Memory Inspector",
        href: "/admin/memory",
        icon: <Icon icon={BrainIcon} className="h-6 w-6" />,
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
