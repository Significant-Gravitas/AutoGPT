import { Navbar } from "@/components/layout/Navbar/Navbar";
import { NetworkStatusMonitor } from "@/services/network-status/NetworkStatusMonitor";
import { PushNotificationProvider } from "@/services/push-notifications/PushNotificationProvider";
import { ReactNode } from "react";
import { AdminImpersonationBanner } from "./admin/components/AdminImpersonationBanner";
import { AutoPilotBridgeProvider } from "@/contexts/AutoPilotBridgeContext";
import { TopUpPromptProvider } from "@/components/layout/TopUpPrompt/TopUpPromptProvider";
import { PaywallGate } from "./PaywallGate/PaywallGate";
import { GlobalSearchOverlay } from "./components/GlobalSearchModal/GlobalSearchOverlay";

export default function PlatformLayout({ children }: { children: ReactNode }) {
  return (
    <AutoPilotBridgeProvider>
      <main className="flex h-screen w-full flex-col">
        <NetworkStatusMonitor />
        <PushNotificationProvider />
        <Navbar />
        <AdminImpersonationBanner />
        <GlobalSearchOverlay />
        <section className="flex-1">
          <TopUpPromptProvider>
            <PaywallGate>{children}</PaywallGate>
          </TopUpPromptProvider>
        </section>
      </main>
    </AutoPilotBridgeProvider>
  );
}
