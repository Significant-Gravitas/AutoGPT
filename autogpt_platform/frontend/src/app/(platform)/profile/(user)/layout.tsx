import * as React from "react";
import { Sidebar } from "@/components/__legacy__/Sidebar";
import {
  IconDashboardLayout,
  IconIntegrations,
  IconProfile,
  IconSliders,
  IconCoin,
} from "@/components/__legacy__/ui/icons";
import { KeyIcon } from "lucide-react";

export default function Layout({ children }: { children: React.ReactNode }) {
  const sidebarLinkGroups = [
    {
      links: [
        {
          text: "Creator Dashboard",
          href: "/profile/dashboard",
          icon: <IconDashboardLayout className="h-6 w-6" />,
        },
        ...(process.env.NEXT_PUBLIC_SHOW_BILLING_PAGE === "true"
          ? [
              {
                text: "Billing",
                href: "/profile/credits",
                icon: <IconCoin className="h-6 w-6" />,
              },
            ]
          : []),
        {
          text: "Integrations",
          href: "/profile/integrations",
          icon: <IconIntegrations className="h-6 w-6" />,
        },
        {
          text: "API Keys",
          href: "/profile/api_keys",
          icon: <KeyIcon className="h-6 w-6" />,
        },
        {
          text: "Profile",
          href: "/profile",
          icon: <IconProfile className="h-6 w-6" />,
        },
        {
          text: "Settings",
          href: "/profile/settings",
          icon: <IconSliders className="h-6 w-6" />,
        },
      ],
    },
  ];

  return (
    <div className="flex min-h-screen w-full max-w-[1360px] flex-col lg:flex-row">
      <Sidebar linkGroups={sidebarLinkGroups} />
      <div className="flex-1 pl-4">{children}</div>
    </div>
  );
}
