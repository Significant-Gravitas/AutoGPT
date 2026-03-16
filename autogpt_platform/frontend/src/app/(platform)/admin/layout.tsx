"use client";

import { Sidebar } from "@/components/__legacy__/Sidebar";
import {
  UsersIcon,
  CurrencyDollarSimpleIcon,
  UserPlusIcon,
  MagnifyingGlassIcon,
  FileTextIcon,
  SlidersHorizontalIcon,
} from "@phosphor-icons/react";

const sidebarLinkGroups = [
  {
    links: [
      {
        text: "Marketplace Management",
        href: "/admin/marketplace",
        icon: <UsersIcon size={24} />,
      },
      {
        text: "User Spending",
        href: "/admin/spending",
        icon: <CurrencyDollarSimpleIcon size={24} />,
      },
      {
        text: "Beta Invites",
        href: "/admin/users",
        icon: <UserPlusIcon size={24} />,
      },
      {
        text: "User Impersonation",
        href: "/admin/impersonation",
        icon: <MagnifyingGlassIcon size={24} />,
      },
      {
        text: "Execution Analytics",
        href: "/admin/execution-analytics",
        icon: <FileTextIcon size={24} />,
      },
      {
        text: "Admin User Management",
        href: "/admin/settings",
        icon: <SlidersHorizontalIcon size={24} />,
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
