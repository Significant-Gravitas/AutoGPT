import type { ReactNode } from "react";
import { Link } from "@/components/atoms/Link/Link";
import { Text } from "@/components/atoms/Text/Text";
import { PRIVACY_POLICY_URL, TERMS_OF_USE_URL } from "@/lib/legal";

export function AuthTermsNotice({ children }: { children?: ReactNode }) {
  return (
    <Text variant="body" tone="muted" className="mt-6 text-center">
      By continuing you agree to our{" "}
      <Link href={TERMS_OF_USE_URL} variant="secondary" isExternal>
        Terms of Use
      </Link>{" "}
      and{" "}
      <Link href={PRIVACY_POLICY_URL} variant="secondary" isExternal>
        Privacy Policy
      </Link>
      .{children}
    </Text>
  );
}
