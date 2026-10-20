import {
  parseSidebarCookie,
  SIDEBAR_COOKIE_NAME,
} from "@/components/ui/sidebar-cookie";
import { NetworkStatusMonitor } from "@/services/network-status/NetworkStatusMonitor";
import { cookies } from "next/headers";
import { PushNotificationProvider } from "@/services/push-notifications/PushNotificationProvider";
import { ReactNode, Suspense } from "react";
import { AutoPilotBridgeProvider } from "@/contexts/AutoPilotBridgeContext";
import { CopilotNotificationsListener } from "./copilot/components/CopilotNotificationsListener/CopilotNotificationsListener";
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
      {/* Reads useSearchParams(), which Next.js requires to be wrapped. */}
      <Suspense fallback={null}>
        <CopilotNotificationsListener />
      </Suspense>
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
