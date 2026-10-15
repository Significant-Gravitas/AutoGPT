import { NetworkStatusMonitor } from "@/services/network-status/NetworkStatusMonitor";
import { PushNotificationProvider } from "@/services/push-notifications/PushNotificationProvider";
import { ReactNode } from "react";
import { AutoPilotBridgeProvider } from "@/contexts/AutoPilotBridgeContext";
import { PlatformChrome } from "./PlatformChrome/PlatformChrome";
import { ProActivationProvider } from "@/services/pro-activation/ProActivationProvider";
import { TrialStatusSync } from "@/services/trials/TrialStatusSync";

export default function PlatformLayout({ children }: { children: ReactNode }) {
  return (
    <AutoPilotBridgeProvider>
      <NetworkStatusMonitor />
      <PushNotificationProvider />
      <TrialStatusSync />
      <ProActivationProvider>
        <PlatformChrome>{children}</PlatformChrome>
      </ProActivationProvider>
    </AutoPilotBridgeProvider>
  );
}
