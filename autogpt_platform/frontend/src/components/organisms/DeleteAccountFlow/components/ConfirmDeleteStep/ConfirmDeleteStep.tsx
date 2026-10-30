import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { Text } from "@/components/atoms/Text/Text";
import { useId } from "react";
import type { DeletionSource } from "../../helpers";
import { StepFooter } from "../StepFooter/StepFooter";
import { ConsequenceList } from "./components/ConsequenceList";
import { useConfirmDeleteStep } from "./useConfirmDeleteStep";

interface Props {
  source: DeletionSource;
  onClose: () => void;
  onBusyChange: (busy: boolean) => void;
}

export function ConfirmDeleteStep({ source, onClose, onBusyChange }: Props) {
  const labelID = useId();
  const inputID = useId();
  const {
    email,
    typedEmail,
    setTypedEmail,
    consequences,
    canDelete,
    isDeleting,
    handleSubmit,
  } = useConfirmDeleteStep({ source, onBusyChange });

  return (
    <form noValidate onSubmit={handleSubmit} className="flex flex-col px-0.5">
      <div className="flex flex-col gap-4">
        <Text variant="body" tone="emphasis">
          This closes your AutoGPT account now and erases it after 14 days.
        </Text>
        <ConsequenceList items={consequences} />
        <div className="flex flex-col gap-2">
          <Text variant="body" id={labelID}>
            Type <span className="font-semibold break-all">{email}</span> to
            confirm
          </Text>
          <Input
            id={inputID}
            label="Your email"
            hideLabel
            aria-labelledby={labelID}
            type="email"
            autoComplete="off"
            autoCapitalize="none"
            autoCorrect="off"
            spellCheck={false}
            value={typedEmail}
            onChange={(event) => setTypedEmail(event.target.value)}
            disabled={isDeleting}
            className="scroll-mb-20 text-base lg:text-sm [@media(max-height:300px)]:scroll-mb-0"
            wrapperClassName="mb-0"
          />
          <Text variant="small" tone="muted">
            You&apos;ll be signed out right away. Changed your mind? Sign in
            within 14 days and choose Keep my account.
          </Text>
        </div>
      </div>
      <StepFooter>
        <Button
          type="button"
          variant="outline"
          onClick={onClose}
          disabled={isDeleting}
          className="flex-1 lg:flex-none"
        >
          Keep account
        </Button>
        <Button
          type="submit"
          variant="destructive"
          disabled={!canDelete}
          loading={isDeleting}
          className="flex-1 lg:flex-none"
        >
          Delete account
        </Button>
      </StepFooter>
    </form>
  );
}
