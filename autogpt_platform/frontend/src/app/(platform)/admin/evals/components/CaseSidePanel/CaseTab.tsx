import {
  Cancel01Icon,
  Target02Icon,
  Tick02Icon,
  UserCheck01Icon,
} from "@hugeicons/core-free-icons";

import {
  BlockNote,
  BlockRow,
} from "@/app/(platform)/copilot/components/WorkArea/blocks";
import {
  InfoGroup,
  InfoList,
  InfoPill,
} from "@/app/(platform)/copilot/components/EmptySession/components/ExpertInfoPanel/components/InfoParts";
import type { EvalCase } from "@/app/api/__generated__/models/evalCase";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";

import { behaviourLabel, categoryLabel, statusOf } from "../../helpers";
import { CaseTrail } from "./CaseTrail";
import { VERDICT_TONES, VERDICT_WORDS } from "./verdicts";

interface Props {
  evalCase: EvalCase;
}

/** One case: what was asked, what was expected against what Otto did, what
 *  the reviewers said and the checks, then the turns. */
export function CaseTab({ evalCase }: Props) {
  const status = statusOf(evalCase);
  const expected = evalCase.expected;
  const matched =
    !evalCase.observed || expected?.behaviour === evalCase.observed;
  const problem = evalCase.error
    ? `The run broke: ${evalCase.error}`
    : evalCase.skipped
      ? `Skipped: ${evalCase.skipped}`
      : null;

  return (
    <div
      aria-label={`Case: ${evalCase.case_id}`}
      className="flex flex-col gap-5"
    >
      <div className="flex flex-col gap-1 px-1">
        <Text variant="small" tone="quiet">
          {categoryLabel(evalCase.category)}
          {evalCase.world ? ` · ${evalCase.world.replaceAll("_", " ")}` : ""}
        </Text>
        <Text variant="body-medium" as="h2" tone="ink" className="text-pretty">
          {evalCase.title || evalCase.case_id}
        </Text>
        <div className="flex items-center gap-1.5 pt-1">
          <InfoPill tone={VERDICT_TONES[status]}>
            {VERDICT_WORDS[status]}
          </InfoPill>
          {evalCase.score !== null && evalCase.score !== undefined ? (
            <InfoPill>{`Score ${Math.round(evalCase.score)}`}</InfoPill>
          ) : null}
        </div>
      </div>

      {problem ? (
        <BlockNote>
          <Text variant="inbox" tone="error">
            {problem}
          </Text>
        </BlockNote>
      ) : null}

      <InfoGroup label="What was asked">
        <BlockNote>
          <Text variant="inbox" tone="ink" className="whitespace-pre-wrap">
            {evalCase.request || "Not saved for this run."}
          </Text>
        </BlockNote>
      </InfoGroup>

      {expected ? (
        <InfoGroup
          label={
            expected.behaviour && evalCase.observed
              ? "Expected and what Otto did"
              : "Expected"
          }
        >
          {expected.behaviour ? (
            <InfoList>
              <BlockRow
                tile={<Icon icon={Target02Icon} size={14} tone="inherit" />}
                title={behaviourLabel(expected.behaviour)}
                subtitle="Expected"
              />
              {evalCase.observed ? (
                <BlockRow
                  tile={
                    <Icon
                      icon={matched ? Tick02Icon : Cancel01Icon}
                      size={14}
                      tone={matched ? "success" : "error"}
                    />
                  }
                  title={behaviourLabel(evalCase.observed)}
                  subtitle="Otto did"
                />
              ) : null}
            </InfoList>
          ) : null}
          <Lines label="Must" lines={expected.must ?? []} />
          <Lines label="Must not" lines={expected.must_not ?? []} />
          {expected.human_reply ? (
            <BlockNote>
              <Text variant="small" tone="quiet">
                A good answer
              </Text>
              <Text variant="inbox" tone="ink" className="whitespace-pre-wrap">
                {expected.human_reply}
              </Text>
              {expected.human_why ? (
                <Text variant="small" tone="quiet">
                  Why: {expected.human_why}
                </Text>
              ) : null}
            </BlockNote>
          ) : null}
        </InfoGroup>
      ) : null}

      <Reviewers evalCase={evalCase} />
      <CaseTrail evalCase={evalCase} />
    </div>
  );
}

function Lines({ label, lines }: { label: string; lines: string[] }) {
  if (!lines.length) return null;
  return (
    <BlockNote>
      <Text variant="small" tone="quiet">
        {label}
      </Text>
      <ul className="flex list-disc flex-col gap-1 pl-4">
        {lines.map((line) => (
          <li key={line}>
            <Text variant="inbox" tone="ink" className="text-pretty">
              {line}
            </Text>
          </li>
        ))}
      </ul>
    </BlockNote>
  );
}

function Reviewers({ evalCase }: { evalCase: EvalCase }) {
  const ratings = evalCase.ratings ?? [];
  if (!ratings.length) return null;
  return (
    <InfoGroup label="What the reviewers said" count={ratings.length}>
      {ratings.map((rating) => (
        <BlockNote key={rating.reviewer}>
          <div className="flex items-center justify-between gap-2">
            <span className="flex min-w-0 items-center gap-1.5 text-foreground-text">
              <Icon icon={UserCheck01Icon} size={14} tone="inherit" />
              <Text
                variant="small"
                weight="medium"
                as="span"
                tone="ink"
                className="truncate"
              >
                {rating.label === "code" ? "code checks" : rating.label}
              </Text>
            </span>
            <InfoPill tone={VERDICT_TONES[rating.verdict]}>
              {`${VERDICT_WORDS[rating.verdict]} · ${Math.round(rating.score)}`}
            </InfoPill>
          </div>
          {(rating.reasons ?? []).length ? (
            <ul className="flex list-disc flex-col gap-1 pl-4">
              {(rating.reasons ?? []).map((reason) => (
                <li key={reason}>
                  <Text variant="inbox" tone="ink" className="text-pretty">
                    {reason}
                  </Text>
                </li>
              ))}
            </ul>
          ) : (
            <Text variant="inbox" tone="quiet">
              No reasons given.
            </Text>
          )}
        </BlockNote>
      ))}
      {evalCase.detail === "report" ? (
        <div className="px-1">
          <Text variant="small" tone="quiet">
            This run&apos;s report kept two reasons per reviewer.
          </Text>
        </div>
      ) : null}
    </InfoGroup>
  );
}
