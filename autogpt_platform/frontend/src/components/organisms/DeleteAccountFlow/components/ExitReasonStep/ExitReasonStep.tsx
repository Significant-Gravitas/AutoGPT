import { QuestionOptionList } from "@/app/(platform)/copilot/components/ChainActionCard/QuestionOptionList";
import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { Text } from "@/components/atoms/Text/Text";
import { useId } from "react";
import type { DeletionSource } from "../../helpers";
import { StepFooter } from "../StepFooter/StepFooter";
import { AlternativePanel } from "./components/AlternativePanel/AlternativePanel";
import { EXIT_REASON_LABELS, MAX_FEEDBACK_TEXT_LENGTH } from "./helpers";
import { useExitReasonStep } from "./useExitReasonStep";

interface Props {
  origin: DeletionSource;
  onClose: () => void;
  onContinue: () => void;
}

export function ExitReasonStep({ origin, onClose, onContinue }: Props) {
  const labelID = useId();
  const textID = useId();
  const {
    reasonLabel,
    placeholder,
    text,
    setText,
    panel,
    isActionPending,
    selectReason,
    runAction,
    skip,
    submit,
  } = useExitReasonStep({ origin, onClose, onContinue });

  return (
    <div className="flex flex-col px-0.5">
      <div className="flex flex-col gap-4">
        <Text variant="body" tone="muted">
          Optional. You can skip straight to deleting, and your answer
          isn&apos;t linked to your account.
        </Text>
        <span id={labelID} className="sr-only">
          Why are you leaving?
        </span>
        <QuestionOptionList
          options={EXIT_REASON_LABELS}
          value={reasonLabel}
          labelId={labelID}
          focusActiveOption={false}
          onChange={selectReason}
          onSubmit={submit}
          editable={false}
          required={false}
          selectedContent={
            panel ? (
              <AlternativePanel
                panel={panel}
                isActionPending={isActionPending}
                onAction={runAction}
              />
            ) : null
          }
        />
        <div role="status" className="sr-only">
          {panel?.copy}
        </div>
        <Input
          id={textID}
          type="textarea"
          rows={1}
          label="Tell us more"
          hideLabel
          placeholder={placeholder}
          maxLength={MAX_FEEDBACK_TEXT_LENGTH}
          value={text}
          onChange={(event) => setText(event.target.value)}
          // A textarea so the placeholder wraps at 320px; the overrides beat
          // the atom's own 14px, one-line textarea classes.
          className="!min-h-16 resize-none scroll-mb-20 !text-base lg:!min-h-[2.875rem] lg:!text-sm [@media(max-height:300px)]:scroll-mb-0"
          wrapperClassName="mb-0"
        />
      </div>
      <StepFooter>
        <Button
          variant="outline"
          onClick={skip}
          className="flex-1 lg:flex-none"
        >
          Skip
        </Button>
        <Button onClick={submit} className="flex-1 lg:flex-none">
          Continue
        </Button>
      </StepFooter>
    </div>
  );
}
