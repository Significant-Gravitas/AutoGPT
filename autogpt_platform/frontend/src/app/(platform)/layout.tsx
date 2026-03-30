"use client";

import { AppSidebar } from "@/components/layout/AppSidebar/AppSidebar";
import { SidebarDynamicContent } from "@/components/layout/AppSidebar/SidebarDynamicContent";
import { Navbar } from "@/components/layout/Navbar/Navbar";
import { SidebarProvider } from "@/components/ui/sidebar";
import { NetworkStatusMonitor } from "@/services/network-status/NetworkStatusMonitor";
import { usePathname } from "next/navigation";
import { ReactNode, useEffect, useRef } from "react";
import { AdminImpersonationBanner } from "./admin/components/AdminImpersonationBanner";

export default function PlatformLayout({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  const scrollRef = useRef<HTMLElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: 0 });
  }, [pathname]);

  return (
    <SidebarProvider defaultOpen={true}>
      <AppSidebar dynamicContent={<SidebarDynamicContent />} />
      <main className="flex h-screen w-full flex-col overflow-hidden">
        <NetworkStatusMonitor />
        <Navbar />
        <AdminImpersonationBanner />
        <section ref={scrollRef} className="flex-1 overflow-y-auto">
          {children}
        </section>
      </main>
    </SidebarProvider>
  );
}
