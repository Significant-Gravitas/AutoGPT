"use client";

import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { WaitlistErrorContent } from "@/components/auth/WaitlistErrorContent";
import { CodeEntry } from "./components/CodeEntry/CodeEntry";
import { ExpiredSignIn } from "./components/ExpiredSignIn/ExpiredSignIn";
import { useSignInCode } from "./useSignInCode";

interface Props {
  next: string | null;
  onBack: () => void;
}

export function SignInCode({ next, onBack }: Props) {
  const signIn = useSignInCode();

  if (signIn.isLoading) {
    return (
      <div className="flex w-full justify-center py-16">
        <LoadingSpinner />
      </div>
    );
  }
  if (signIn.screen === "waitlisted") {
    return <WaitlistErrorContent onBackToLogin={onBack} />;
  }
  if (signIn.screen === "expired") return <ExpiredSignIn onBack={onBack} />;
  return (
    <CodeEntry
      next={next}
      email={signIn.email}
      providerName={signIn.providerName}
      resendAt={signIn.resendAt}
      onExpired={signIn.handleExpired}
      onWaitlisted={signIn.handleWaitlisted}
      onBack={onBack}
    />
  );
}
