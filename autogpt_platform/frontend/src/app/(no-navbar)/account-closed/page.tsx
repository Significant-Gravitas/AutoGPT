"use client";

import { Button } from "@/components/atoms/Button/Button";
import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { Text } from "@/components/atoms/Text/Text";
import { OpenAgain } from "./components/OpenAgain";
import { PageCard } from "./components/PageCard";
import { StillClosedHelp } from "./components/StillClosedHelp";
import { useAccountClosedPage } from "./useAccountClosedPage";

export default function AccountClosedPage() {
  const {
    view,
    eraseDate,
    isReady,
    isKeeping,
    isLeaving,
    keepAccount,
    continueToApp,
    signOut,
  } = useAccountClosedPage();

  if (view === "loading" || view === "leaving") {
    return (
      <main className="flex min-h-screen items-center justify-center px-4">
        <LoadingSpinner size="large" />
      </main>
    );
  }

  if (view === "open") {
    return (
      <OpenAgain
        isReady={isReady}
        isLeaving={isLeaving}
        onContinue={continueToApp}
      />
    );
  }

  return (
    <PageCard title="Your account is closed">
      {view === "error" && (
        <Text variant="body">
          We couldn&apos;t load your account. Reload the page to try again.
        </Text>
      )}
      {view === "too_late" && (
        <>
          <Text variant="body">
            The 14 days to keep your account are over, and everything in it is
            being erased.
          </Text>
          <StillClosedHelp variant="too_late" />
        </>
      )}
      {view === "closed" && (
        <>
          <Text variant="body">
            We&apos;ll erase your account and everything in it on{" "}
            <span className="font-semibold">{eraseDate}</span>. Changed your
            mind? Keep your account and use AutoGPT again.
          </Text>
          <Text variant="small" tone="muted">
            Your plan, connected apps, API keys, schedules and share links
            stopped and your Marketplace listings and reviews were removed when
            you closed your account, and notification emails are off. If you
            keep it, subscribe again, set up what you need and switch emails
            back on in Settings.
          </Text>
        </>
      )}
      <div className="flex flex-col-reverse gap-2 sm:flex-row sm:justify-end">
        <Button
          type="button"
          variant="secondary"
          onClick={signOut}
          disabled={isKeeping || isLeaving}
        >
          Sign out
        </Button>
        {view === "closed" && (
          <Button
            type="button"
            onClick={keepAccount}
            loading={isKeeping}
            disabled={isLeaving}
          >
            Keep my account
          </Button>
        )}
      </div>
    </PageCard>
  );
}
