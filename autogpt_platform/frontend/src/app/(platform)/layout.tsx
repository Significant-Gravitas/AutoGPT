import { NetworkStatusMonitor } from "@/services/network-status/NetworkStatusMonitor";
import { PushNotificationProvider } from "@/services/push-notifications/PushNotificationProvider";
import { cookies } from "next/headers";
import { ReactNode } from "react";
import { AutoPilotBridgeProvider } from "@/contexts/AutoPilotBridgeContext";
import { LayoutHintProvider } from "./PlatformChrome/components/LayoutHintProvider/LayoutHintProvider";
import { readLayoutHint } from "./PlatformChrome/helpers";
import { PlatformChrome } from "./PlatformChrome/PlatformChrome";

export default async function PlatformLayout({
  children,
}: {
  children: ReactNode;
}) {
  // Which shell the layout flag last resolved to for this browser, so the
  // server paints it straight away instead of the chrome swapping after the
  // flag vendor answers (see usePlatformChrome).
  const layoutHint = readLayoutHint((await cookies()).getAll());

  return (
    <AutoPilotBridgeProvider>
      <NetworkStatusMonitor />
      <PushNotificationProvider />
      <LayoutHintProvider hint={layoutHint}>
        <PlatformChrome>{children}</PlatformChrome>
      </LayoutHintProvider>
    </AutoPilotBridgeProvider>
  );
}
