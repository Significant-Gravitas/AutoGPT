"use client";

import { useReportClientUrl } from "./useReportClientUrl";
import { usePushNotifications } from "./usePushNotifications";

export function PushNotificationProvider() {
  usePushNotifications();
  useReportClientUrl();
  return null;
}
