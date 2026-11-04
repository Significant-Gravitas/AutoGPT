import type { Contract } from "@/app/api/__generated__/models/contract";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import {
  CRITERION_GLYPH,
  criteriaMeta,
  evidenceById,
  looseEvidence,
} from "../../helpers";
import { PanelSection } from "../PanelSection/PanelSection";
import { ProofLine } from "../ProofLine/ProofLine";

interface Props {
  taskId: string;
  contract: Contract;
}

/** The checks that say the task is done, each with the proof behind it. */
export function DoneWhen({ taskId, contract }: Props) {
  const criteria = contract.done_when ?? [];
  const loose = looseEvidence(contract);
  if (!criteria.length && !loose.length) return null;
  const byId = evidenceById(contract);

  return (
    <PanelSection
      label={criteria.length ? "Done when" : "Proof"}
      meta={criteria.length ? criteriaMeta(criteria) : undefined}
    >
      {criteria.length ? (
        <ul className="flex flex-col gap-2.5">
          {criteria.map((criterion) => {
            const glyph = CRITERION_GLYPH[criterion.status ?? "pending"];
            const proof = (criterion.evidence_ids ?? [])
              .map((id) => byId.get(id))
              .filter((e) => e !== undefined);
            return (
              <li key={criterion.id} className="flex items-start gap-2.5">
                <Icon
                  icon={glyph.icon}
                  tone={glyph.tone}
                  size={14}
                  aria-label={glyph.label}
                  className="mt-0.5 shrink-0"
                />
                <div className="flex min-w-0 flex-1 flex-col gap-1">
                  <Text
                    variant="inbox"
                    tone={criterion.status === "skipped" ? "faint" : "ink"}
                  >
                    {criterion.text}
                  </Text>
                  {proof.map((evidence) => (
                    <ProofLine
                      key={evidence.id}
                      taskId={taskId}
                      evidence={evidence}
                    />
                  ))}
                </div>
              </li>
            );
          })}
        </ul>
      ) : null}
      {loose.length ? (
        <div className="flex flex-col gap-1.5">
          {criteria.length ? (
            <Text variant="small" tone="faint">
              Also kept
            </Text>
          ) : null}
          {loose.map((evidence) => (
            <ProofLine key={evidence.id} taskId={taskId} evidence={evidence} />
          ))}
        </div>
      ) : null}
    </PanelSection>
  );
}
