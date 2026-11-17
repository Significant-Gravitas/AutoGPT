import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import type { ConnectingProvider } from "@/lib/auth/connect-attempt";

interface Props {
  provider: ConnectingProvider;
  disabled?: boolean;
  onSignIn: () => void;
}

const PROVIDER_NAMES: Record<ConnectingProvider, string> = {
  google: "Google",
  microsoft: "Microsoft",
};

export function SignInWithoutConnecting({
  provider,
  disabled,
  onSignIn,
}: Props) {
  return (
    <div className="flex flex-col items-center gap-1 text-center">
      <Text variant="small" tone="muted">
        Couldn&apos;t connect your email and files? Some organizations
        don&apos;t allow it. You can sign in now and connect later in Settings.
      </Text>
      <Button
        type="button"
        variant="ghost"
        size="small"
        disabled={disabled}
        onClick={onSignIn}
      >
        Continue with {PROVIDER_NAMES[provider]} without connecting
      </Button>
    </div>
  );
}
