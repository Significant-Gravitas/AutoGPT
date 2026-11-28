"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { MailValidation01Icon } from "@hugeicons/core-free-icons";
import { useFormStatus } from "react-dom";
import { confirmEmail } from "../actions";

interface Props {
  email: string | null;
}

export function ConfirmEmail({ email }: Props) {
  return (
    <main className="flex min-h-screen items-center justify-center px-4 py-8">
      <form
        action={confirmEmail}
        className="flex w-full max-w-md flex-col items-center rounded-large border border-border-base bg-surface p-6 text-center"
      >
        <div className="mb-6 flex size-14 items-center justify-center rounded-full bg-accent-surface">
          <Icon
            icon={MailValidation01Icon}
            tone="brand"
            className="size-7"
            aria-hidden
          />
        </div>
        <Text variant="h3" as="h1">
          Confirm your email
        </Text>
        <Text variant="body" tone="secondary" className="mt-3">
          We&apos;ll verify{" "}
          {email ? (
            <span className="font-medium break-all text-foreground-base">
              {email}
            </span>
          ) : (
            "your email"
          )}{" "}
          and sign you in to AutoGPT.
        </Text>
        <VerifyButton />
      </form>
    </main>
  );
}

function VerifyButton() {
  const { pending } = useFormStatus();
  return (
    <Button type="submit" loading={pending} className="mt-8 w-full">
      Verify and sign in
    </Button>
  );
}
