"use client";

import { CSSProperties, ReactNode } from "react";

import { TourSidebar } from "@/app/(public)/tour/chat/components/TourSidebar/TourSidebar";
import { AppSidebar } from "@/components/layout/AppSidebar/AppSidebar";
import { AppSidebarToggleProvider } from "@/components/layout/AppSidebar/components/AppSidebarTrigger/AppSidebarTrigger";
import { TopUpPromptProvider } from "@/components/layout/TopUpPrompt/TopUpPromptProvider";
import { ChangelogPopup } from "@/components/molecules/ChangelogPopup/ChangelogPopup";
import {
  getPersistedSidebarOpen,
  SidebarInset,
  SidebarProvider,
  SidebarTrigger,
} from "@/components/ui/sidebar";
import { cn } from "@/lib/utils";

import { AdminImpersonationBanner } from "../admin/components/AdminImpersonationBanner";
import { GlobalSearchOverlay } from "../components/GlobalSearchModal/GlobalSearchOverlay";
import { PaywallGate } from "../PaywallGate/PaywallGate";
import { ArtifactsPanelNavCollapse } from "./components/ArtifactsPanelNavCollapse/ArtifactsPanelNavCollapse";
import { BuilderSidebarAutoClose } from "./components/BuilderSidebarAutoClose/BuilderSidebarAutoClose";
import { InsetHeaderActions } from "./components/InsetHeaderActions/InsetHeaderActions";
import { InsetHeaderSidebarTrigger } from "./components/InsetHeaderSidebarTrigger/InsetHeaderSidebarTrigger";
import { InsetHeaderTitle } from "./components/InsetHeaderTitle/InsetHeaderTitle";
import { usePlatformChrome } from "./usePlatformChrome";

interface Props {
  children: ReactNode;
  // The request's sidebar cookie, for a server render.
  persistedSidebarOpen?: boolean;
}

export function PlatformChrome({ children, persistedSidebarOpen }: Props) {
  const {
    showAppSidebar,
    showChangelog,
    showTourSidebar,
    overlayInsetHeader,
    hasInsetHeaderTitle,
    isBuilderRoute,
  } = usePlatformChrome();

  const content = (
    <TopUpPromptProvider>
      <PaywallGate>{children}</PaywallGate>
    </TopUpPromptProvider>
  );

  // Logged-out marketplace visitors get public navigation and a trial CTA.
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

  if (showAppSidebar) {
    // The browser's cookie stays current across client navigation; the layout's does not.
    const openOutsideBuilder =
      getPersistedSidebarOpen() ?? persistedSidebarOpen ?? false;
    return (
      <SidebarProvider
        persistState
        defaultOpen={!isBuilderRoute && openOutsideBuilder}
        style={{ "--sidebar-width": "18.25rem" } as CSSProperties}
      >
        <BuilderSidebarAutoClose openOutsideBuilder={openOutsideBuilder} />
        <ArtifactsPanelNavCollapse />
        <AppSidebar />
        {showChangelog ? <ChangelogPopup /> : null}
        <SidebarInset className="bg-[#f9f9f9]">
          <header
            className={cn(
              "flex shrink-0 items-center pb-4 pt-6",
              // Overlay mode (copilot): the header floats above the content
              // instead of reserving vertical space, so the chat scrolls to
              // the viewport top underneath it.
              overlayInsetHeader
                ? "pointer-events-none absolute inset-x-0 top-0 z-40"
                : "relative z-10",
              !overlayInsetHeader && !hasInsetHeaderTitle && "md:hidden",
            )}
          >
            <div className="mx-auto flex w-full max-w-7xl items-center gap-2 px-6 md:px-8">
              <InsetHeaderSidebarTrigger />
              <InsetHeaderTitle />
              <InsetHeaderActions />
            </div>
          </header>
          <AdminImpersonationBanner />
          <GlobalSearchOverlay />
          <AppSidebarToggleProvider>
            <section className="flex-1">{content}</section>
          </AppSidebarToggleProvider>
        </SidebarInset>
      </SidebarProvider>
    );
  }

  // Settings and admin render their own sidebar shells (with a Back link);
  // the unauthenticated routes (reset-password, auth errors) get a bare frame.
  return (
    <main className="flex h-screen w-full flex-col">
      <AdminImpersonationBanner />
      <GlobalSearchOverlay />
      <section className="flex-1">{content}</section>
    </main>
  );
}
