"use client";

import { useAdminImpersonation } from "./useAdminImpersonation";

export function AdminImpersonationBanner() {
  const { isImpersonating, impersonatedUserId, stopImpersonating } =
    useAdminImpersonation();

  if (!isImpersonating) {
    return null;
  }

  return (
    <div className="mb-4 rounded-md border border-warning bg-warning-surface p-4 text-warning-foreground">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <strong className="font-semibold">
            ⚠️ ADMIN IMPERSONATION ACTIVE
          </strong>
          <span>
            You are currently acting as user:{" "}
            <code className="rounded bg-warning-surface px-1 font-mono text-sm">
              {impersonatedUserId}
            </code>
          </span>
        </div>
        <button
          onClick={stopImpersonating}
          className="ml-4 flex h-8 items-center rounded-md border border-warning-border bg-transparent px-3 text-sm hover:bg-warning-surface"
        >
          Stop Impersonation
        </button>
      </div>
    </div>
  );
}
