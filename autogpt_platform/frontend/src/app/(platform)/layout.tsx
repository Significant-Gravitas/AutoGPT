import { NetworkStatusMonitor } from "@/services/network-status/NetworkStatusMonitor";
import { PushNotificationProvider } from "@/services/push-notifications/PushNotificationProvider";
import { ReactNode } from "react";
import { AutoPilotBridgeProvider } from "@/contexts/AutoPilotBridgeContext";
import { PlatformChrome } from "./PlatformChrome/PlatformChrome";
import { ProActivationProvider } from "@/services/pro-activation/ProActivationProvider";

export default function PlatformLayout({ children }: { children: ReactNode }) {
  return (
    <AutoPilotBridgeProvider>
      <NetworkStatusMonitor />
      <PushNotificationProvider />
      <ProActivationProvider>
        <PlatformChrome>{children}</PlatformChrome>
      </ProActivationProvider>
    </AutoPilotBridgeProvider>
  );
}
