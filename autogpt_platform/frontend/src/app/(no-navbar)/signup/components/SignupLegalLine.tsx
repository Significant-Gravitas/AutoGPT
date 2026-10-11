import {
  Link,
  linkBaseClasses,
  linkVariantClasses,
} from "@/components/atoms/Link/Link";
import { Text } from "@/components/atoms/Text/Text";
import { PRIVACY_POLICY_URL, TERMS_OF_USE_URL } from "@/lib/legal";
import { cn } from "@/lib/utils";

interface Props {
  optedOut: boolean;
  onToggle: () => void;
  disabled?: boolean;
}

export function SignupLegalLine({ optedOut, onToggle, disabled }: Props) {
  return (
    <Text variant="body" className="mt-6 text-center !text-slate-500">
      By continuing you agree to our{" "}
      <Link href={TERMS_OF_USE_URL} variant="secondary" isExternal>
        Terms of Use
      </Link>{" "}
      and{" "}
      <Link href={PRIVACY_POLICY_URL} variant="secondary" isExternal>
        Privacy Policy
      </Link>
      .{" "}
      <span aria-live="polite">
        {optedOut ? (
          <span className="text-slate-950">
            You won&apos;t get marketing emails.
          </span>
        ) : (
          "We may email you product updates and offers;"
        )}
      </span>{" "}
      <button
        type="button"
        onClick={onToggle}
        disabled={disabled}
        className={cn(
          linkBaseClasses,
          linkVariantClasses.secondary,
          "rounded-sm focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-2",
          "disabled:cursor-not-allowed disabled:opacity-50",
        )}
      >
        {optedOut ? "Undo" : "opt out"}
      </button>
      .
    </Text>
  );
}
