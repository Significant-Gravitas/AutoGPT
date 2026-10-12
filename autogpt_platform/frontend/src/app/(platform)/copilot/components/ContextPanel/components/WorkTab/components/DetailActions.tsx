"use client";

import {
  ArrowReloadHorizontalIcon,
  ArrowRight01Icon,
  Cancel01Icon,
  Message01Icon,
} from "@hugeicons/core-free-icons";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import type {
  ChatDelegation,
  LiveDelegationStatus,
} from "../../../../../delegations";
import type { LiveDelegation } from "../../../../../useDelegationLive";
import { useDelegationAnswer } from "../../../../../useDelegationAnswer";
import { DelegationQuestionBox } from "../../../../DelegationQuestion/DelegationQuestionBox";
import { isBudgetError } from "../helpers";
import type { useDelegationControls } from "../useDelegationControls";
import { DetailOutcome } from "./DetailOutcome";
import { DetailSection } from "./DetailSection";

interface Props {
  delegation: ChatDelegation;
  live: LiveDelegation;
  href: string | null;
  controls: ReturnType<typeof useDelegationControls>;
}

const PILL = "rounded-full";

function OpenThread({ href, primary }: { href: string; primary?: boolean }) {
  return (
    <Button
      as="NextLink"
      href={href}
      variant={primary ? "primary" : "secondary"}
      size="xs"
      className={PILL}
      rightIcon={<Icon icon={ArrowRight01Icon} size={12} />}
    >
      Open thread
    </Button>
  );
}

function CancelButton({
  controls,
  wide,
}: Pick<Props, "controls"> & { wide?: boolean }) {
  return (
    <Button
      variant="secondary"
      size="xs"
      className={wide ? `${PILL} w-full` : PILL}
      leadingIcon={wide ? undefined : Cancel01Icon}
      loading={controls.isCancelling}
      onClick={() => void controls.cancel()}
    >
      Cancel
    </Button>
  );
}

function QuestionActions({
  delegation,
  live,
}: Pick<Props, "delegation" | "live">) {
  const { sendAnswer, isSending } = useDelegationAnswer();
  const question = live.question;
  const subSessionId = delegation.subSessionId;
  if (!question || !subSessionId) return null;
  return (
    <DetailSection title={`${live.expert.name} asks`}>
      <div className="rounded-xl border border-amber-200 bg-amber-50 p-3">
        <DelegationQuestionBox
          question={question}
          options={live.questionOptions}
          expertName={live.expert.name}
          variant="panel"
          isSending={isSending}
          onSend={(answer) => void sendAnswer(subSessionId, answer, question)}
        />
      </div>
    </DetailSection>
  );
}

function statusActions(status: LiveDelegationStatus, props: Props) {
  const { delegation, live, href, controls } = props;
  const name = live.expert.name;
  switch (status) {
    case "queued":
      return (
        <DetailSection title="Status">
          <p className="text-sm text-zinc-700">
            {name} starts as soon as they are free.
          </p>
          {delegation.subSessionId && <CancelButton controls={controls} wide />}
        </DetailSection>
      );
    case "running":
      return (
        <DetailSection title="Controls">
          <div className="flex flex-wrap gap-2">
            {href && <OpenThread href={href} />}
            {controls.canAskOtto && (
              <Button
                variant="secondary"
                size="xs"
                className={PILL}
                leadingIcon={Message01Icon}
                onClick={controls.nudge}
              >
                Nudge
              </Button>
            )}
            {delegation.subSessionId && <CancelButton controls={controls} />}
          </div>
        </DetailSection>
      );
    case "needs-input":
      return <QuestionActions delegation={delegation} live={live} />;
    case "completed":
      return (
        <DetailSection title="What came back">
          <DetailOutcome response={live.response} files={delegation.files} />
          <div className="flex flex-wrap gap-2">
            {href && <OpenThread href={href} primary />}
            {controls.canAskOtto && (
              <Button
                variant="secondary"
                size="xs"
                className={PILL}
                leadingIcon={ArrowReloadHorizontalIcon}
                onClick={controls.redelegate}
              >
                Re-delegate
              </Button>
            )}
          </div>
        </DetailSection>
      );
    case "failed":
      return (
        <DetailSection title="What to do">
          {delegation.error && (
            <p className="text-sm text-red-700">{delegation.error}</p>
          )}
          <div className="flex flex-wrap gap-2">
            {controls.canAskOtto && isBudgetError(delegation.error) && (
              <Button
                variant="primary"
                size="xs"
                className={PILL}
                onClick={controls.raiseBudget}
              >
                Raise budget and retry
              </Button>
            )}
            {controls.canAskOtto && (
              <Button
                variant="secondary"
                size="xs"
                className={PILL}
                leadingIcon={ArrowReloadHorizontalIcon}
                onClick={controls.retry}
              >
                Retry
              </Button>
            )}
            {href && <OpenThread href={href} />}
          </div>
        </DetailSection>
      );
    default:
      return href ? (
        <div className="flex flex-wrap gap-2">
          <OpenThread href={href} />
        </div>
      ) : null;
  }
}

/** The one block of the detail that changes with the hand-off's state. */
export function DetailActions(props: Props) {
  return statusActions(props.live.status, props);
}
