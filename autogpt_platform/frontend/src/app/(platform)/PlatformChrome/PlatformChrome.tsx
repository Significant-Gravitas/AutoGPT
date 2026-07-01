"use client";

import { CSSProperties, ReactNode } from "react";

import { AppSidebar } from "@/components/layout/AppSidebar/AppSidebar";
import { Navbar } from "@/components/layout/Navbar/Navbar";
import { TopUpPromptProvider } from "@/components/layout/TopUpPrompt/TopUpPromptProvider";
import {
  SidebarInset,
  SidebarProvider,
  SidebarTrigger,
} from "@/components/ui/sidebar";

import { AdminImpersonationBanner } from "../admin/components/AdminImpersonationBanner";
import { GlobalSearchOverlay } from "../components/GlobalSearchModal/GlobalSearchOverlay";
import { PaywallGate } from "../PaywallGate/PaywallGate";
import { InsetHeaderActions } from "./components/InsetHeaderActions/InsetHeaderActions";
import { usePlatformChrome } from "./usePlatformChrome";

interface Props {
  children: ReactNode;
}

export function PlatformChrome({ children }: Props) {
  const { showNewLayout } = usePlatformChrome();

  const content = (
    <TopUpPromptProvider>
      <PaywallGate>{children}</PaywallGate>
    </TopUpPromptProvider>
  );

  if (showNewLayout) {
    return (
      <SidebarProvider style={{ "--sidebar-width": "19rem" } as CSSProperties}>
        <AppSidebar />
        <SidebarInset className="bg-[#F8F8F9]">
          <header className="flex h-12 shrink-0 items-center justify-end gap-2 px-4">
            <div className="mr-auto md:hidden">
              <SidebarTrigger />
            </div>
            <InsetHeaderActions />
          </header>
          <AdminImpersonationBanner />
          <GlobalSearchOverlay />
          <section className="flex-1">{content}</section>
        </SidebarInset>
      </SidebarProvider>
    );
  }

  return (
    <main className="flex h-screen w-full flex-col">
      <Navbar />
      <AdminImpersonationBanner />
      <GlobalSearchOverlay />
      <section className="flex-1">{content}</section>
    </main>
  );
}
