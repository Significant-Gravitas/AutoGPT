"use client";

import { DecisionRequestAction } from "@/app/api/__generated__/models/decisionRequestAction";
import { SkillVersionSummary } from "@/app/api/__generated__/models/skillVersionSummary";
import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { Text } from "@/components/atoms/Text/Text";
import {
  stepChanges,
  stripFrontmatter,
} from "@/services/skill-learning/helpers";

interface Props {
  proposal: SkillVersionSummary;
  current: SkillVersionSummary | null;
  draft: string;
  isBusy: boolean;
  onDraftChange: (value: string) => void;
  onDecide: (versionId: string, action: DecisionRequestAction) => void;
}

export function DecisionView({
  proposal,
  current,
  draft,
  isBusy,
  onDraftChange,
  onDecide,
}: Props) {
  const changes = stepChanges(current?.body ?? "", proposal.body ?? "");
  return (
    <section
      aria-label="Needs your decision"
      className="flex flex-col gap-3 rounded-xl border border-amber-200 bg-amber-50/60 p-3"
    >
      <Text variant="small-medium" tone="primary">
        Needs your decision
      </Text>
      <Text variant="small" tone="secondary">
        {proposal.summary || proposal.description} — {proposal.state_reason}
      </Text>
      <ul className="flex flex-col gap-2" aria-label="Proposed changes by step">
        {changes.map((change) => (
          <li key={change.step} className="rounded-md bg-white p-2">
            <Text variant="small-medium" tone="primary">
              {change.step}
            </Text>
            {change.before ? (
              <Text variant="small" tone="muted" className="line-through">
                {change.before}
              </Text>
            ) : null}
            <Text variant="small" tone="primary">
              {change.after ?? "Removed"}
            </Text>
          </li>
        ))}
      </ul>
      <Input
        id={`decision-edit-${proposal.id}`}
        label="Edited alternative (optional)"
        labelVariant="small-medium"
        type="textarea"
        rows={5}
        value={draft}
        placeholder={stripFrontmatter(proposal.body ?? "").slice(0, 120)}
        onChange={(event) => onDraftChange(event.target.value)}
      />
      <div className="flex flex-wrap gap-2">
        <Button
          variant="secondary"
          size="small"
          disabled={isBusy}
          onClick={() => onDecide(proposal.id, "keep_current")}
        >
          Keep current
        </Button>
        <Button
          variant="primary"
          size="small"
          disabled={isBusy}
          onClick={() => onDecide(proposal.id, "apply")}
        >
          Apply change
        </Button>
        <Button
          variant="ghost"
          size="small"
          disabled={isBusy || !draft.trim()}
          onClick={() => onDecide(proposal.id, "apply_edited")}
        >
          Apply edited alternative
        </Button>
      </div>
    </section>
  );
}
