"use client";

import { CSSProperties, ReactNode } from "react";

import { TourSidebar } from "@/app/(public)/tour/chat/components/TourSidebar/TourSidebar";
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
import { InsetHeaderTitle } from "./components/InsetHeaderTitle/InsetHeaderTitle";
import { usePlatformChrome } from "./usePlatformChrome";

interface Props {
  children: ReactNode;
}

export function PlatformChrome({ children }: Props) {
  const { showNewLayout, showTourSidebar } = usePlatformChrome();

  const content = (
    <TopUpPromptProvider>
      <PaywallGate>{children}</PaywallGate>
    </TopUpPromptProvider>
  );

  // Logged-out marketplace visitors browse with the tour demo sidebar as an
  // upsell — clicking a demo session takes them into /tour/chat.
  if (showTourSidebar) {
    return (
      <SidebarProvider style={{ "--sidebar-width": "19rem" } as CSSProperties}>
        <TourSidebar variant="marketplace" />
        <SidebarInset className="bg-[#f9f9f9]">
          <div className="flex shrink-0 items-center px-4 pt-4 md:hidden">
            <SidebarTrigger />
          </div>
          <section className="flex-1">{content}</section>
        </SidebarInset>
      </SidebarProvider>
    );
  }

  if (showNewLayout) {
    return (
      <SidebarProvider style={{ "--sidebar-width": "19rem" } as CSSProperties}>
        <AppSidebar />
        <SidebarInset className="bg-[#f9f9f9]">
          <header className="relative flex shrink-0 items-center pb-4 pt-6">
            <div className="mx-auto flex w-full max-w-7xl items-center gap-2 px-6 md:px-8">
              <div className="md:hidden">
                <SidebarTrigger />
              </div>
              <InsetHeaderTitle />
            </div>
            <div className="absolute inset-y-0 right-4 flex items-center">
              <InsetHeaderActions />
            </div>
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
