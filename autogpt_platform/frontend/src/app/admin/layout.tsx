import { ShoppingBag } from "lucide-react";
import { Sidebar } from "@/components/agptui/Sidebar";
import { Users, DollarSign, LogOut } from "lucide-react";

import { IconSliders } from "@/components/ui/icons";

const sidebarLinkGroups = [
  {
    links: [
      {
        text: "Agent Management",
        href: "/admin/agents",
        icon: <Users className="h-6 w-6" />,
      },
      {
        text: "User Spending",
        href: "/admin/spending",
        icon: <DollarSign className="h-6 w-6" />,
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
    <div className="flex min-h-screen w-screen max-w-[1360px] flex-col lg:flex-row">
      <Sidebar linkGroups={sidebarLinkGroups} />
      <div className="flex-1 pl-4">{children}</div>
    </div>
  );
}
