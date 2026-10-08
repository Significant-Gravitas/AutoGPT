"use client";

import { usePushNotifications } from "./usePushNotifications";
import { useReportClientUrl } from "./useReportClientUrl";
import { useReportNotificationsEnabled } from "./useReportNotificationsEnabled";
import { useNativePush } from "./native/useNativePush";

export function PushNotificationProvider() {
  useNativePush();
  usePushNotifications();
  useReportClientUrl();
  useReportNotificationsEnabled();
  return null;
}
