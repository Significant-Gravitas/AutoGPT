"use client";

import { X } from "lucide-react";
import { useAdminImpersonation } from "@/hooks/useAdminImpersonation";

/**
 * Security banner that appears when an admin is impersonating another user.
 *
 * Displays prominently at the top of the page with warning styling and
 * provides a quick way to stop impersonation.
 */
export function AdminImpersonationBanner() {
  const { isImpersonating, impersonatedUserId, stopImpersonating } =
    useAdminImpersonation();

  if (!isImpersonating) {
    return null;
  }

  return (
    <div className="mb-4 rounded-md border border-amber-500 bg-amber-50 p-4 text-amber-900">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <strong className="font-semibold">
            ⚠️ ADMIN IMPERSONATION ACTIVE
          </strong>
          <span>
            You are currently acting as user:{" "}
            <code className="rounded bg-amber-100 px-1 font-mono text-sm">
              {impersonatedUserId}
            </code>
          </span>
        </div>
        <button
          onClick={stopImpersonating}
          className="ml-4 flex h-8 items-center rounded-md border border-amber-300 bg-transparent px-3 text-sm hover:bg-amber-100"
        >
          <X className="mr-1 h-3 w-3" />
          Stop Impersonation
        </button>
      </div>
    </div>
  );
}
