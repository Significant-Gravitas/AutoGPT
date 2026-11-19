import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { SIGN_IN_CODE_TTL_SECONDS } from "@/lib/auth/sign-in-code";
import type { SignInCodeEnd } from "../../helpers";
import { useResendCode } from "./useResendCode";

interface Props {
  email: string | null;
  resendAt: number;
  onEnded: (end: SignInCodeEnd) => void;
  onResent: () => void;
}

export function ResendCode(props: Props) {
  const { cooldown, isResending, canResend, handleResend } =
    useResendCode(props);

  return (
    <>
      <Text variant="small" tone="muted" className="mt-4">
        The code expires in {SIGN_IN_CODE_TTL_SECONDS / 60} minutes. Can&apos;t
        find it? Check your spam folder.
      </Text>
      <Button
        variant="secondary"
        type="button"
        loading={isResending}
        disabled={!canResend}
        onClick={handleResend}
        className="mt-6 w-full"
      >
        Send a new code
      </Button>
      {/* Beside the button, not in its name, so a screen reader doesn't
          announce the button again every second. */}
      {cooldown > 0 ? (
        <Text variant="small" tone="muted" className="mt-2">
          You can ask for another in {cooldown}s.
        </Text>
      ) : null}
    </>
  );
}
