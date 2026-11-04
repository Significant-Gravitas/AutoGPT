import type { Step } from "@/app/api/__generated__/models/step";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import type { TaskState } from "../../../../data/types";
import { StateDot } from "../../../StateDot/StateDot";
import {
  countSettled,
  PLAN_STEP_GLYPH,
  planMeta,
  ROW_TEXT_TONE,
  STEP_TEXT_TONE,
} from "../../helpers";
import { PanelSection } from "../PanelSection/PanelSection";

interface Props {
  plan: Step[];
  now: string;
  state: TaskState;
}

/** The plan as it runs: each step, with work going on at once as rows. */
export function Progress({ plan, now, state }: Props) {
  if (!plan.length) {
    if (state !== "working" || !now) return null;
    return (
      <PanelSection label="Progress">
        <div className="flex items-start gap-2">
          <StateDot state={state} className="mt-0.5" />
          <Text variant="inbox" tone="text">
            {now}
          </Text>
        </div>
      </PanelSection>
    );
  }

  return (
    <PanelSection label="Progress" meta={planMeta(plan)}>
      <ol className="flex flex-col gap-2">
        {plan.map((step) => {
          const status = step.status ?? "pending";
          const glyph = PLAN_STEP_GLYPH[status];
          const rows = step.rows ?? [];
          return (
            <li key={step.id} className="flex flex-col gap-1.5">
              <div className="flex items-start gap-2.5">
                <Icon
                  icon={glyph.icon}
                  tone={glyph.tone}
                  size={14}
                  spin={status === "running"}
                  aria-label={glyph.label}
                  className="mt-0.5 shrink-0"
                />
                <Text
                  variant="inbox"
                  tone={STEP_TEXT_TONE[status]}
                  className="min-w-0 flex-1"
                >
                  {step.title}
                </Text>
                {rows.length ? (
                  <Text
                    variant="small"
                    tone="faint"
                    tabular
                    className="shrink-0"
                  >
                    {countSettled(rows)} of {rows.length}
                  </Text>
                ) : null}
              </div>
              {rows.length ? (
                <ul className="flex flex-col gap-1 pl-6">
                  {rows.map((row) => {
                    const rowStatus = row.status ?? "pending";
                    const rowGlyph = PLAN_STEP_GLYPH[rowStatus];
                    return (
                      <li key={row.id} className="flex items-center gap-2">
                        <Icon
                          icon={rowGlyph.icon}
                          tone={rowGlyph.tone}
                          size={12}
                          spin={rowStatus === "running"}
                          aria-label={rowGlyph.label}
                          className="shrink-0"
                        />
                        <Text
                          variant="small"
                          tone={ROW_TEXT_TONE[rowStatus]}
                          className="min-w-0 flex-1 truncate"
                        >
                          {row.title}
                        </Text>
                      </li>
                    );
                  })}
                </ul>
              ) : null}
            </li>
          );
        })}
      </ol>
    </PanelSection>
  );
}
