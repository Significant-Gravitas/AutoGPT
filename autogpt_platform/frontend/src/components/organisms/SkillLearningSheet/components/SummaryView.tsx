"use client";

import { SkillLearningDetail } from "@/app/api/__generated__/models/skillLearningDetail";
import { SkillVersionSummary } from "@/app/api/__generated__/models/skillVersionSummary";
import { Badge } from "@/components/atoms/Badge/Badge";
import { Button } from "@/components/atoms/Button/Button";
import { Switch } from "@/components/atoms/Switch/Switch";
import { Text } from "@/components/atoms/Text/Text";
import { reuseLabel, versionLabel } from "@/services/skill-learning/helpers";
import { formatDistanceToNow } from "date-fns";
import { OutcomeReportRequestOutcome } from "@/app/api/__generated__/models/outcomeReportRequestOutcome";
import { SkillPolicyRequest } from "@/app/api/__generated__/models/skillPolicyRequest";

interface Props {
  detail: SkillLearningDetail;
  version: SkillVersionSummary | null;
  isBusy: boolean;
  onPolicy: (data: SkillPolicyRequest) => void;
  onEdit: () => void;
  onReport: (versionId: string, outcome: OutcomeReportRequestOutcome) => void;
}

function stateVariant(state: string) {
  if (state === "ready") return "success" as const;
  if (state === "blocked_content" || state === "invalidated")
    return "error" as const;
  if (state === "needs_decision" || state === "paused")
    return "warning" as const;
  return "info" as const;
}

export function SummaryView({
  detail,
  version,
  isBusy,
  onPolicy,
  onEdit,
  onReport,
}: Props) {
  const isCurrent = version?.id === detail.current_version?.id;
  const badgeState = version && !isCurrent ? version.state : detail.state;
  const badgeLabel =
    version && !isCurrent ? version.state_label : detail.state_label;
  return (
    <div className="flex flex-col gap-5">
      <div className="flex flex-wrap items-center gap-2">
        <span data-testid="version-state">
          <Badge variant={stateVariant(badgeState)}>{badgeLabel}</Badge>
        </span>
        {version ? (
          <Text variant="small" tone="muted" as="span">
            {versionLabel(version)} ·{" "}
            {formatDistanceToNow(new Date(version.created_at), {
              addSuffix: true,
            })}
            {isCurrent ? "" : " · not the current version"}
          </Text>
        ) : null}
      </div>
      {version && !isCurrent ? (
        <Text variant="small" tone="muted">
          Skill now: {detail.state_label}. The controls below apply to the skill
          as a whole, not to this historical version.
        </Text>
      ) : null}
      {version ? (
        <>
          <Text variant="body" tone="primary">
            {version.summary || version.description}
          </Text>
          <section aria-label="Evidence">
            <Text variant="small-medium" tone="primary">
              Evidence
            </Text>
            <ul className="mt-1 flex flex-col gap-1">
              {(version.evidence ?? []).map((item) => (
                <li key={`${item.kind}-${item.ref}-${item.label}`}>
                  <Text variant="small" tone="secondary">
                    {item.label}
                  </Text>
                </li>
              ))}
              <li>
                <Text
                  variant="small"
                  tone="secondary"
                  data-testid="reuse-label"
                >
                  {reuseLabel(version)}
                </Text>
              </li>
            </ul>
          </section>
          {(version.limits ?? []).length > 0 ? (
            <section aria-label="Limits">
              <Text variant="small-medium" tone="primary">
                Limits
              </Text>
              <ul className="mt-1 list-disc pl-4">
                {(version.limits ?? []).map((limit) => (
                  <li key={limit}>
                    <Text variant="small" tone="secondary">
                      {limit}
                    </Text>
                  </li>
                ))}
              </ul>
            </section>
          ) : null}
          {version.blocked_pattern_class ? (
            <Text variant="small" tone="danger">
              Blocked by content check: {version.blocked_pattern_class} at{" "}
              {version.blocked_step}. Replace the value with a typed input.
            </Text>
          ) : null}
        </>
      ) : (
        <Text variant="small" tone="muted">
          No version has been published yet.
        </Text>
      )}
      <section aria-label="Learning controls" className="flex flex-col gap-3">
        <label className="flex items-center justify-between gap-3">
          <Text variant="small" tone="primary">
            Automatic improvements
          </Text>
          <Switch
            checked={detail.policy.auto_improve}
            disabled={isBusy}
            aria-label="Automatic improvements"
            onCheckedChange={(checked) => onPolicy({ auto_improve: checked })}
          />
        </label>
        <div className="flex flex-wrap gap-2">
          <Button
            variant="secondary"
            size="small"
            disabled={isBusy}
            onClick={() =>
              onPolicy({ learning_paused: !detail.policy.learning_paused })
            }
          >
            {detail.policy.learning_paused
              ? "Resume learning"
              : "Pause learning"}
          </Button>
          <Button
            variant="secondary"
            size="small"
            disabled={isBusy}
            onClick={() => onPolicy({ use_paused: !detail.policy.use_paused })}
          >
            {detail.policy.use_paused
              ? "Resume using this skill"
              : "Stop using this skill"}
          </Button>
        </div>
        <Text variant="small" tone="muted">
          Pause learning stops future automatic changes only; stop using removes
          the skill from retrieval. Each is separate.
        </Text>
        <div className="flex flex-wrap gap-2">
          <Button
            variant="secondary"
            size="small"
            disabled={isBusy || !isCurrent}
            title={
              isCurrent
                ? undefined
                : "Editing starts from the current version. Restore this version first to build on it."
            }
            onClick={onEdit}
          >
            Edit skill
          </Button>
          {version ? (
            <>
              <Button
                variant="ghost"
                size="small"
                disabled={isBusy}
                onClick={() => onReport(version.id, "succeeded")}
              >
                Report: it worked
              </Button>
              <Button
                variant="ghost"
                size="small"
                disabled={isBusy}
                onClick={() => onReport(version.id, "failed")}
              >
                Report: it failed
              </Button>
            </>
          ) : null}
        </div>
        <Text variant="small" tone="muted">
          Edit skill applies your correction now
          {isCurrent ? "" : " (available on the current version)"}. Report an
          outcome adds evidence for a later review without changing the skill.
        </Text>
      </section>
    </div>
  );
}
