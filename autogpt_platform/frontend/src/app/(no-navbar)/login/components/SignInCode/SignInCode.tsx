"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Input } from "@/components/atoms/Input/Input";
import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { Text } from "@/components/atoms/Text/Text";
import { WaitlistErrorContent } from "@/components/auth/WaitlistErrorContent";
import { SIGN_IN_CODE_LENGTH } from "@/lib/auth/sign-in-code";
import { Mail01Icon } from "@hugeicons/core-free-icons";
import { SIGN_IN_CODE_INPUT_ID } from "./helpers";
import { useSignInCode } from "./useSignInCode";

interface Props {
  next: string | null;
  onBack: () => void;
}

export function SignInCode({ next, onBack }: Props) {
  const {
    screen,
    isLoading,
    email,
    providerName,
    code,
    error,
    isVerifying,
    isResending,
    cooldown,
    canResend,
    handleCodeChange,
    handleSubmit,
    handleResend,
  } = useSignInCode({ next });

  if (isLoading) {
    return (
      <div className="flex w-full justify-center py-16">
        <LoadingSpinner />
      </div>
    );
  }

  if (screen === "waitlisted") {
    return <WaitlistErrorContent onBackToLogin={onBack} />;
  }

  if (screen === "expired") {
    return (
      <div className="flex w-full flex-col items-center text-center">
        <Text variant="h3" as="h1">
          This sign-in has expired
        </Text>
        <Text variant="body" tone="secondary" className="mt-3">
          Sign in again and we&apos;ll email you a new code.
        </Text>
        <Button
          variant="primary"
          type="button"
          onClick={onBack}
          className="mt-8 w-full"
        >
          Back to log in
        </Button>
      </div>
    );
  }

  return (
    <div className="flex w-full flex-col items-center text-center">
      <div className="mb-6 flex size-14 items-center justify-center rounded-full bg-accent-surface">
        <Icon icon={Mail01Icon} tone="brand" className="size-7" />
      </div>

      <Text variant="h3" as="h1">
        Check your email
      </Text>

      {email && providerName ? (
        <Text variant="body" tone="secondary" className="mt-3">
          We sent a {SIGN_IN_CODE_LENGTH}-digit code to{" "}
          <span className="font-medium break-all text-foreground-base">
            {email}
          </span>
          . Enter it to finish signing in with {providerName}.
        </Text>
      ) : (
        <Text variant="body" tone="secondary" className="mt-3">
          We sent a {SIGN_IN_CODE_LENGTH}-digit code to your email address.
          Enter it to finish signing in.
        </Text>
      )}

      <form
        onSubmit={handleSubmit}
        className="mt-8 flex w-full flex-col gap-1 text-left"
      >
        <Input
          id={SIGN_IN_CODE_INPUT_ID}
          label="Code"
          placeholder="123456"
          inputMode="numeric"
          autoComplete="one-time-code"
          maxLength={SIGN_IN_CODE_LENGTH + 2}
          value={code}
          onChange={(event) => handleCodeChange(event.target.value)}
          error={error ?? undefined}
          className="w-full"
        />
        <Button
          variant="primary"
          type="submit"
          loading={isVerifying}
          disabled={!code.trim()}
          className="mt-4 w-full"
        >
          Continue
        </Button>
      </form>

      <Text variant="small" tone="muted" className="mt-4">
        The code expires in 10 minutes. Can&apos;t find it? Check your spam
        folder.
      </Text>

      <Button
        variant="secondary"
        type="button"
        loading={isResending}
        disabled={!canResend}
        onClick={handleResend}
        className="mt-6 w-full"
      >
        {cooldown > 0 ? `Send a new code in ${cooldown}s` : "Send a new code"}
      </Button>

      <div className="mt-6 inline-flex items-center justify-center gap-1">
        <Text variant="body-medium" tone="muted">
          Wrong account?
        </Text>
        <Button type="button" variant="link" size="inline" onClick={onBack}>
          Back to log in
        </Button>
      </div>
    </div>
  );
}
