"use client";

import { Button } from "@/components/atoms/Button/Button";
import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { Text } from "@/components/atoms/Text/Text";
import { useAccountClosedPage } from "./useAccountClosedPage";

export default function AccountClosedPage() {
  const { eraseDate, isLoading, isError, isKeeping, keepAccount, signOut } =
    useAccountClosedPage();

  if (isLoading) {
    return (
      <main className="flex min-h-screen items-center justify-center px-4">
        <LoadingSpinner size="large" />
      </main>
    );
  }

  return (
    <main className="flex min-h-screen items-center justify-center px-4 py-8">
      <div className="flex w-full max-w-md flex-col gap-4 rounded-large border border-border-base bg-surface p-6">
        <Text variant="h3" as="h1">
          Your account is closed
        </Text>
        {isError ? (
          <Text variant="body">
            We couldn&apos;t load your account. Reload the page to try again.
          </Text>
        ) : (
          <>
            <Text variant="body">
              We&apos;ll erase your account and everything in it on{" "}
              <span className="font-semibold">{eraseDate}</span>. Changed your
              mind? Keep your account and use AutoGPT again.
            </Text>
            <Text variant="small" tone="muted">
              Your plan, connected apps, schedules and share links stopped when
              you closed your account. If you keep it, subscribe again and set
              those up again.
            </Text>
          </>
        )}
        <div className="flex flex-col-reverse gap-2 sm:flex-row sm:justify-end">
          <Button
            type="button"
            variant="secondary"
            onClick={signOut}
            disabled={isKeeping}
          >
            Sign out
          </Button>
          {!isError && (
            <Button type="button" onClick={keepAccount} loading={isKeeping}>
              Keep my account
            </Button>
          )}
        </div>
      </div>
    </main>
  );
}
