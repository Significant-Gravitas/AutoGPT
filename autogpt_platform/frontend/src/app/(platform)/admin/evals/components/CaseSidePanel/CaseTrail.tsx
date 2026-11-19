import {
  Cancel01Icon,
  Tick02Icon,
  Wrench01Icon,
} from "@hugeicons/core-free-icons";

import {
  BlockNote,
  BlockRow,
} from "@/app/(platform)/copilot/components/WorkArea/blocks";
import {
  InfoGroup,
  InfoList,
  InfoMeta,
} from "@/app/(platform)/copilot/components/EmptySession/components/ExpertInfoPanel/components/InfoParts";
import type { EvalCase } from "@/app/api/__generated__/models/evalCase";
import type { EvalTurn } from "@/app/api/__generated__/models/evalTurn";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";

interface Props {
  evalCase: EvalCase;
}

/** The case's checks, then what happened turn by turn, then the feature's output. */
export function CaseTrail({ evalCase }: Props) {
  const checks = [...(evalCase.checks ?? [])].sort(
    (a, b) => Number(a.passed) - Number(b.passed),
  );
  const turns = evalCase.turns ?? [];
  return (
    <>
      {checks.length ? (
        <InfoGroup label="Checks" count={checks.length}>
          <InfoList>
            {checks.map((check) => (
              <BlockRow
                key={check.name}
                tile={
                  <Icon
                    icon={check.passed ? Tick02Icon : Cancel01Icon}
                    size={14}
                    tone={check.passed ? "success" : "error"}
                  />
                }
                title={check.name}
                subtitle={check.detail || null}
                wrap
                trailing={
                  <InfoMeta>{check.passed ? "Passed" : "Failed"}</InfoMeta>
                }
              />
            ))}
          </InfoList>
          {evalCase.detail === "report" ? (
            <div className="px-1">
              <Text variant="small" tone="quiet">
                This run&apos;s report kept only the failed checks.
              </Text>
            </div>
          ) : null}
        </InfoGroup>
      ) : null}
      {turns.map((turn, i) => (
        <Turn key={i} turn={turn} index={i} />
      ))}
      {evalCase.output ? (
        <InfoGroup label="What the feature produced">
          <BlockNote>
            <Text
              variant="small-mono"
              as="pre"
              className="max-h-80 overflow-auto whitespace-pre-wrap"
            >
              {evalCase.output}
            </Text>
          </BlockNote>
        </InfoGroup>
      ) : null}
    </>
  );
}

function Turn({ turn, index }: { turn: EvalTurn; index: number }) {
  const calls = turn.tool_calls ?? [];
  const ended = `ended by ${turn.ended_by?.replaceAll("_", " ")}`;
  return (
    <InfoGroup
      label={`Turn ${index + 1}`}
      action={<InfoMeta>{ended}</InfoMeta>}
    >
      {calls.length ? (
        <InfoList>
          {calls.map((call, i) => (
            <BlockRow
              key={i}
              tile={
                <Icon
                  icon={
                    call.refused || call.failed ? Cancel01Icon : Wrench01Icon
                  }
                  size={14}
                  tone={call.refused || call.failed ? "error" : "inherit"}
                />
              }
              title={call.name}
              subtitle={
                call.refused ||
                (call.arguments !== "{}" ? call.arguments : null)
              }
              wrap
              trailing={<InfoMeta>{callFlags(call)}</InfoMeta>}
            />
          ))}
        </InfoList>
      ) : null}
      {(turn.questions ?? []).map((q) => (
        <BlockNote key={q.question}>
          <Text variant="small" tone="quiet">
            Asked{q.reason ? ` · ${q.reason.replaceAll("_", " ")}` : ""}
          </Text>
          <Text variant="inbox" tone="ink" className="text-pretty">
            {q.question}
          </Text>
        </BlockNote>
      ))}
      {turn.reply ? (
        <BlockNote>
          <Text variant="small" tone="quiet">
            Otto said
          </Text>
          <Text variant="inbox" tone="ink" className="whitespace-pre-wrap">
            {turn.reply}
          </Text>
        </BlockNote>
      ) : null}
    </InfoGroup>
  );
}

function callFlags(call: NonNullable<EvalTurn["tool_calls"]>[number]): string {
  if (call.refused) return "refused";
  if (call.failed) return "failed";
  if (call.held) return "held";
  if (call.wrote) return "wrote";
  return call.service ?? "";
}
