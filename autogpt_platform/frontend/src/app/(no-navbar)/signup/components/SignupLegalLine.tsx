import {
  Link,
  linkBaseClasses,
  linkFocusClasses,
  linkVariantClasses,
} from "@/components/atoms/Link/Link";
import { Text } from "@/components/atoms/Text/Text";
import { PRIVACY_POLICY_URL, TERMS_OF_USE_URL } from "@/lib/legal";
import { cn } from "@/lib/utils";

interface Props {
  optedOut: boolean;
  onToggle: () => void;
}

export function SignupLegalLine({ optedOut, onToggle }: Props) {
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
        className={cn(
          linkBaseClasses,
          linkVariantClasses.secondary,
          linkFocusClasses,
        )}
      >
        {optedOut ? "Undo" : "opt out"}
      </button>
      .
    </Text>
  );
}
