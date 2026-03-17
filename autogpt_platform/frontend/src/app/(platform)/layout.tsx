"use client";

import { AppSidebar } from "@/components/layout/AppSidebar/AppSidebar";
import { SidebarDynamicContent } from "@/components/layout/AppSidebar/SidebarDynamicContent";
import { Navbar } from "@/components/layout/Navbar/Navbar";
import { SidebarProvider } from "@/components/ui/sidebar";
import { NetworkStatusMonitor } from "@/services/network-status/NetworkStatusMonitor";
import { ReactNode } from "react";
import { AdminImpersonationBanner } from "./admin/components/AdminImpersonationBanner";

export default function PlatformLayout({ children }: { children: ReactNode }) {
  return (
    <SidebarProvider defaultOpen={true}>
      <AppSidebar dynamicContent={<SidebarDynamicContent />} />
      <main className="flex h-screen w-full flex-col">
        <NetworkStatusMonitor />
        <Navbar />
        <AdminImpersonationBanner />
        <section className="flex-1 overflow-hidden">{children}</section>
      </main>
    </SidebarProvider>
  );
}
