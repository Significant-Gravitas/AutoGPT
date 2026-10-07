"use client";

import { Button } from "@/components/atoms/Button/Button";
import { useAdminImpersonation } from "./useAdminImpersonation";

export function AdminImpersonationBanner() {
  const { isImpersonating, impersonatedUserId, stopImpersonating } =
    useAdminImpersonation();

  if (!isImpersonating) {
    return null;
  }

  return (
    <div className="mb-4 rounded-md border border-yellow-500 bg-yellow-50 p-4 text-yellow-900">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <strong className="font-semibold">
            ⚠️ ADMIN IMPERSONATION ACTIVE
          </strong>
          <span>
            You are currently acting as user:{" "}
            <code className="rounded bg-yellow-100 px-1 font-mono text-sm">
              {impersonatedUserId}
            </code>
          </span>
        </div>
        <Button
          variant="outline"
          size="small"
          onClick={stopImpersonating}
          className="ml-4"
        >
          Stop Impersonation
        </Button>
      </div>
    </div>
  );
}
