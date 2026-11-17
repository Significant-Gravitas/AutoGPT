import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";

interface Props {
  onBack: () => void;
}

export function ExpiredSignIn({ onBack }: Props) {
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
