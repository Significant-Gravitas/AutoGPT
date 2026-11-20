import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { SIGN_IN_CODE_LENGTH } from "@/lib/auth/sign-in-code";
import { Mail01Icon } from "@hugeicons/core-free-icons";
import { useState } from "react";
import type { SignInCodeEnd } from "../../helpers";
import { CodeForm } from "../CodeForm/CodeForm";
import { ResendCode } from "../ResendCode/ResendCode";

interface Props {
  next: string | null;
  email: string | null;
  providerName: string | null;
  resendAt: number;
  onEnded: (end: SignInCodeEnd) => void;
  onWaitlisted: () => void;
  onBack: () => void;
}

export function CodeEntry(props: Props) {
  // A new code starts the form over.
  const [codesSent, setCodesSent] = useState(1);

  return (
    <div className="flex w-full flex-col items-center text-center">
      <div className="mb-6 flex size-14 items-center justify-center rounded-full bg-accent-surface">
        <Icon icon={Mail01Icon} tone="brand" className="size-7" />
      </div>
      <Text variant="h3" as="h1">
        Check your email
      </Text>
      <Text variant="body" tone="secondary" className="mt-3">
        We sent a {SIGN_IN_CODE_LENGTH}-digit code to{" "}
        {props.email ? (
          <span className="font-medium break-all text-foreground-base">
            {props.email}
          </span>
        ) : (
          "your email address"
        )}
        . Enter it to finish signing in
        {props.providerName ? ` with ${props.providerName}` : ""}.
      </Text>
      <CodeForm
        key={codesSent}
        next={props.next}
        onEnded={props.onEnded}
        onWaitlisted={props.onWaitlisted}
      />
      <ResendCode
        email={props.email}
        resendAt={props.resendAt}
        onEnded={props.onEnded}
        onResent={() => setCodesSent((count) => count + 1)}
      />
      <div className="mt-6 inline-flex items-center justify-center gap-1">
        <Text variant="body-medium" tone="muted">
          Wrong account?
        </Text>
        <Button
          type="button"
          variant="link"
          size="inline"
          onClick={props.onBack}
        >
          Back to log in
        </Button>
      </div>
    </div>
  );
}
