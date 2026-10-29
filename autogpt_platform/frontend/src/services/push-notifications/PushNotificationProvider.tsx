"use client";

import { usePushNotifications } from "./usePushNotifications";
import { useReportClientUrl } from "./useReportClientUrl";
import { useReportNotificationsEnabled } from "./useReportNotificationsEnabled";
import { useSyncNotificationsEnabled } from "./useSyncNotificationsEnabled";

export function PushNotificationProvider() {
  useSyncNotificationsEnabled();
  usePushNotifications();
  useReportClientUrl();
  useReportNotificationsEnabled();
  return null;
}
