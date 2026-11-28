"use client";

import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";

import { OAuthGrantRow } from "../OAuthGrantRow/OAuthGrantRow";
import { RevokeOAuthGrantDialog } from "../RevokeOAuthGrantDialog/RevokeOAuthGrantDialog";
import { useOAuthGrantList } from "./useOAuthGrantList";

const LIST_CLASSES =
  "flex w-full flex-col divide-y divide-border-base overflow-hidden rounded-2xl border border-border-base bg-surface";

export function OAuthGrantList() {
  const {
    grants,
    isLoading,
    isError,
    error,
    refetch,
    isEmpty,
    revokeTarget,
    requestRevoke,
    closeRevokeDialog,
  } = useOAuthGrantList();

  if (isLoading) {
    return (
      <div
        role="status"
        aria-label="Loading OAuth apps"
        className={LIST_CLASSES}
      >
        {[0, 1, 2].map((i) => (
          <div key={i} className="flex items-center gap-4 px-5 py-4">
            <Skeleton className="h-10 w-10 rounded-xl" />
            <div className="flex flex-col gap-2">
              <Skeleton className="h-4 w-40" />
              <Skeleton className="h-3 w-56" />
            </div>
          </div>
        ))}
      </div>
    );
  }
  if (isError) {
    const message = error instanceof Error ? error.message : undefined;
    return (
      <ErrorCard
        context="OAuth apps"
        responseError={message ? { message } : undefined}
        onRetry={() => {
          refetch();
        }}
      />
    );
  }
  if (isEmpty) {
    return (
      <div className="flex flex-col items-center gap-1 px-6 py-10 text-center">
        <Text variant="large-medium" as="span">
          No apps have access
        </Text>
        <Text variant="body" tone="muted" className="max-w-[400px]">
          When you let an app sign in with your AutoGPT account, it shows up
          here, so you can take its access back.
        </Text>
      </div>
    );
  }

  return (
    <>
      <ul className={LIST_CLASSES}>
        {grants.map((grant) => (
          <OAuthGrantRow
            key={grant.application_id}
            grant={grant}
            onRevoke={() => requestRevoke(grant)}
          />
        ))}
      </ul>
      {revokeTarget && (
        <RevokeOAuthGrantDialog
          grant={revokeTarget}
          onOpenChange={(open) => {
            if (!open) closeRevokeDialog();
          }}
        />
      )}
    </>
  );
}
