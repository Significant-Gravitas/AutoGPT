import { Sidebar } from "@/components/__legacy__/Sidebar";
import { Users, DollarSign } from "lucide-react";

import { IconSliders } from "@/components/__legacy__/ui/icons";

const sidebarLinkGroups = [
  {
    links: [
      {
        text: "Marketplace Management",
        href: "/admin/marketplace",
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
    <div className="flex min-h-screen w-full flex-col lg:flex-row">
      <Sidebar linkGroups={sidebarLinkGroups} />
      <div className="flex-1 pl-4">{children}</div>
    </div>
  );
}
