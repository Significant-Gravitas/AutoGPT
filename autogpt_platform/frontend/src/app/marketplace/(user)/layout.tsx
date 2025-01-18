import * as React from "react";
import { Sidebar } from "@/components/agptui/Sidebar";

export default function Layout({ children }: { children: React.ReactNode }) {
  const sidebarLinkGroups = [
    {
      links: [
        { text: "Creator Dashboard", href: "/marketplace/dashboard" },
        { text: "Agent dashboard", href: "/marketplace/agent-dashboard" },
        { text: "Credits", href: "/marketplace/credits" },
        { text: "Integrations", href: "/marketplace/integrations" },
        { text: "API Keys", href: "/marketplace/api_keys" },
        { text: "Profile", href: "/marketplace/profile" },
        { text: "Settings", href: "/marketplace/settings" },
      ],
    },
  ];

  return (
    <div className="flex min-h-screen w-screen max-w-[1360px] flex-col lg:flex-row">
      <Sidebar linkGroups={sidebarLinkGroups} />
      <div className="flex-1 pl-4">{children}</div>
    </div>
  );
}
