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
import { PaywallGate } from "../PaywallGate/PaywallGate";
import { InsetHeaderActions } from "./components/InsetHeaderActions/InsetHeaderActions";
import { usePlatformChrome } from "./usePlatformChrome";

type Props = { children: ReactNode };

export function PlatformChrome({ children }: Props) {
  const { showNewLayout } = usePlatformChrome();

  const content = (
    <TopUpPromptProvider>
      <PaywallGate>{children}</PaywallGate>
    </TopUpPromptProvider>
  );

  if (showNewLayout) {
    return (
      <SidebarProvider
        style={{ "--sidebar-width": "19rem" } as CSSProperties}
      >
        <AppSidebar />
        <SidebarInset className="bg-[#F8F8F9]">
          <header className="flex h-12 shrink-0 items-center justify-between gap-2 px-4">
            <SidebarTrigger />
            <InsetHeaderActions />
          </header>
          <AdminImpersonationBanner />
          <section className="flex-1">{content}</section>
        </SidebarInset>
      </SidebarProvider>
    );
  }

  return (
    <main className="flex h-screen w-full flex-col">
      <Navbar />
      <AdminImpersonationBanner />
      <section className="flex-1">{content}</section>
    </main>
  );
}
