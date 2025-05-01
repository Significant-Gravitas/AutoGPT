import * as React from "react";
import { Sidebar } from "@/components/agptui/Sidebar";
import {
  IconDashboardLayout,
  IconIntegrations,
  IconProfile,
  IconSliders,
  IconCoin,
} from "@/components/ui/icons";
import { KeyIcon } from "lucide-react";

export default function Layout({ children }: { children: React.ReactNode }) {
  const sidebarLinkGroups = [
    {
      links: [
        {
          text: "Creator Dashboard",
          href: "/profile/dashboard",
          icon: <IconDashboardLayout className="h-6 w-6 stroke-[1.25px]" />,
        },
        ...(process.env.NEXT_PUBLIC_SHOW_BILLING_PAGE === "true"
          ? [
              {
                text: "Billing",
                href: "/profile/credits",
                icon: <IconCoin className="h-6 w-6 stroke-[1.25px]" />,
              },
            ]
          : []),
        {
          text: "Integrations",
          href: "/profile/integrations",
          icon: <IconIntegrations className="h-6 w-6 stroke-[1.25px]" />,
        },
        {
          text: "API Keys",
          href: "/profile/api_keys",
          icon: <KeyIcon className="h-6 w-6 stroke-[1.25px]" />,
        },
        {
          text: "Profile",
          href: "/profile",
          icon: <IconProfile className="h-6 w-6 stroke-[1.25px]" />,
        },
        {
          text: "Settings",
          href: "/profile/settings",
          icon: <IconSliders className="h-6 w-6 stroke-[1.25px]" />,
        },
      ],
    },
  ];

  return (
    <div className="flex flex-row gap-14 px-4 pr-10">
      <Sidebar linkGroups={sidebarLinkGroups} />
      <div className="flex-1">{children}</div>
    </div>
  );
}
