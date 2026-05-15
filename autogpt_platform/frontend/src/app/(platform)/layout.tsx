import { Navbar } from "@/components/layout/Navbar/Navbar";
import { NetworkStatusMonitor } from "@/services/network-status/NetworkStatusMonitor";
import { PushNotificationProvider } from "@/services/push-notifications/PushNotificationProvider";
import { ReactNode } from "react";
import { AdminImpersonationBanner } from "./admin/components/AdminImpersonationBanner";
import { AutoPilotBridgeProvider } from "@/contexts/AutoPilotBridgeContext";
import { PaywallGate } from "./PaywallGate/PaywallGate";

export default function PlatformLayout({ children }: { children: ReactNode }) {
  return (
    <AutoPilotBridgeProvider>
      <main className="flex h-screen w-full flex-col">
        <NetworkStatusMonitor />
        <PushNotificationProvider />
        <Navbar />
        <AdminImpersonationBanner />
        <section className="flex-1">
          <PaywallGate>{children}</PaywallGate>
        </section>
      </main>
    </AutoPilotBridgeProvider>
  );
}
