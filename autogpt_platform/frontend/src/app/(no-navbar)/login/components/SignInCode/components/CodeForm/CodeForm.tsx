import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { SIGN_IN_CODE_LENGTH } from "@/lib/auth/sign-in-code";
import { SIGN_IN_CODE_INPUT_ID, type SignInCodeEnd } from "../../helpers";
import { useCodeForm } from "./useCodeForm";

interface Props {
  next: string | null;
  onEnded: (end: SignInCodeEnd) => void;
  onWaitlisted: () => void;
}

export function CodeForm(props: Props) {
  const { code, error, isVerifying, handleCodeChange, handleSubmit } =
    useCodeForm(props);

  return (
    <form
      onSubmit={handleSubmit}
      className="mt-8 flex w-full flex-col gap-1 text-left"
    >
      <Input
        id={SIGN_IN_CODE_INPUT_ID}
        label="Code"
        placeholder="123456"
        inputMode="numeric"
        autoComplete="one-time-code"
        // Room for a space or two typed in the middle; the server drops them.
        maxLength={SIGN_IN_CODE_LENGTH + 2}
        value={code}
        onChange={(event) => handleCodeChange(event.target.value)}
        error={error ?? undefined}
        className="w-full"
      />
      <Button
        variant="primary"
        type="submit"
        loading={isVerifying}
        disabled={!code.trim()}
        className="mt-4 w-full"
      >
        Continue
      </Button>
    </form>
  );
}
