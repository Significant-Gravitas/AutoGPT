import { Sidebar } from "@/components/__legacy__/Sidebar";
import {
  Users,
  CurrencyDollar,
  UserFocus,
  FileText,
  Database,
  Faders,
} from "@phosphor-icons/react";

const sidebarLinkGroups = [
  {
    links: [
      {
        text: "Marketplace Management",
        href: "/admin/marketplace",
        icon: <Users size={24} />,
      },
      {
        text: "User Spending",
        href: "/admin/spending",
        icon: <CurrencyDollar size={24} />,
      },
      {
        text: "User Impersonation",
        href: "/admin/impersonation",
        icon: <UserFocus size={24} />,
      },
      {
        text: "Execution Analytics",
        href: "/admin/execution-analytics",
        icon: <FileText size={24} />,
      },
      {
        text: "Admin User Management",
        href: "/admin/settings",
        icon: <Faders size={24} />,
      },
      {
        text: "Test Data",
        href: "/admin/test-data",
        icon: <Database size={24} />,
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
    <div className="flex min-h-screen w-full flex-col lg:flex-row">
      <Sidebar linkGroups={sidebarLinkGroups} />
      <div className="flex-1 pl-4">{children}</div>
    </div>
  );
}
