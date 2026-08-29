"use client";

import { usePushNotifications } from "./usePushNotifications";
import { useReportClientUrl } from "./useReportClientUrl";
import { useReportNotificationsEnabled } from "./useReportNotificationsEnabled";

export function PushNotificationProvider() {
  usePushNotifications();
  useReportClientUrl();
  useReportNotificationsEnabled();
  return null;
}
