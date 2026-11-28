import {
  linkBaseClasses,
  linkVariantClasses,
} from "@/components/atoms/Link/Link";
import { AuthTermsNotice } from "@/components/auth/AuthTermsNotice";
import { cn } from "@/lib/utils";

interface Props {
  optedOut: boolean;
  onToggle: () => void;
  disabled?: boolean;
}

export function SignupLegalLine({ optedOut, onToggle, disabled }: Props) {
  return (
    <AuthTermsNotice>
      {" "}
      <span aria-live="polite">
        {optedOut ? (
          <span className="text-foreground-base">
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
          "rounded-sm focus:outline-none focus-visible:ring-2 focus-visible:ring-info focus-visible:ring-offset-2",
          "disabled:cursor-not-allowed disabled:opacity-50",
        )}
      >
        {optedOut ? "Undo" : "opt out"}
      </button>
      .
    </AuthTermsNotice>
  );
}
