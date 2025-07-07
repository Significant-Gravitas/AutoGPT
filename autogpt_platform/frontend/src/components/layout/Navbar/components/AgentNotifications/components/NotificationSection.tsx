"use client";

import { ReactNode } from "react";

interface NotificationSectionProps {
  title: string;
  count: number;
  colorClass: string;
  children: ReactNode;
}

export function NotificationSection({
  title,
  count,
  colorClass,
  children,
}: NotificationSectionProps) {
  return (
    <div className="border-b border-gray-100 p-4 dark:border-gray-700">
      <h4 className={`mb-2 text-sm font-medium ${colorClass}`}>
        {title} ({count})
      </h4>
      <div className="space-y-2">{children}</div>
    </div>
  );
}
