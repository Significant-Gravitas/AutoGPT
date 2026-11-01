import {
  parseSidebarCookie,
  SIDEBAR_COOKIE_NAME,
} from "@/components/molecules/Sidebar/sidebar-cookie";
import { NetworkStatusMonitor } from "@/services/network-status/NetworkStatusMonitor";
import { cookies } from "next/headers";
import { PushNotificationProvider } from "@/services/push-notifications/PushNotificationProvider";
import { ReactNode } from "react";
import { AutoPilotBridgeProvider } from "@/contexts/AutoPilotBridgeContext";
import { PlatformChrome } from "./PlatformChrome/PlatformChrome";
import { ProActivationProvider } from "@/services/pro-activation/ProActivationProvider";
import { TrialStatusSync } from "@/services/trials/TrialStatusSync";

export default async function PlatformLayout({
  children,
}: {
  children: ReactNode;
}) {
  const sidebarCookie = (await cookies()).get(SIDEBAR_COOKIE_NAME)?.value;
  return (
    <AutoPilotBridgeProvider>
      <NetworkStatusMonitor />
      <PushNotificationProvider />
      <TrialStatusSync />
      <ProActivationProvider>
        <PlatformChrome
          persistedSidebarOpen={parseSidebarCookie(sidebarCookie)}
        >
          {children}
        </PlatformChrome>
      </ProActivationProvider>
    </AutoPilotBridgeProvider>
  );
}
