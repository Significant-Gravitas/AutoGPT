import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import type { SignInCodeEnd } from "../../helpers";
import { ENDED_SIGN_IN_COPY } from "./helpers";

interface Props {
  end: SignInCodeEnd;
  onBack: () => void;
}

export function EndedSignIn({ end, onBack }: Props) {
  const copy = ENDED_SIGN_IN_COPY[end];

  return (
    <div className="flex w-full flex-col items-center text-center">
      <Text variant="h3" as="h1">
        {copy.title}
      </Text>
      <Text variant="body" tone="secondary" className="mt-3">
        {copy.body}
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
