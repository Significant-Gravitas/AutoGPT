import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import { SpendRing } from "./components/SpendRing";
import { spendSubtitle, type SpendSummary } from "./helpers";

interface Props {
  summary: SpendSummary;
  pendingAction: "approve" | "reject" | null;
  disabled: boolean;
  onDecide: (approved: boolean) => void;
}

export function SpendApprovalCard({
  summary,
  pendingAction,
  disabled,
  onDecide,
}: Props) {
  return (
    <div
      data-slot="spend-approval-card"
      className="flex flex-wrap items-center gap-x-4 gap-y-3 rounded-2xl border border-border-base bg-surface py-3 pl-4 pr-3"
    >
      <SpendRing spent={summary.spent} threshold={summary.threshold}>
        <ExpertAvatar
          name={summary.expertName}
          avatarUrl={summary.avatarUrl}
          size={32}
        />
      </SpendRing>
      <div className="min-w-0 flex-1">
        <Text variant="body-semibold" tone="primary">
          {summary.expertName} has done {summary.spent} credits of work{" "}
          {summary.period}
        </Text>
        <Text variant="small" tone="muted">
          {spendSubtitle(summary)}
        </Text>
      </div>
      <div className="ml-auto flex items-center gap-1">
        <Button
          variant="ghost"
          size="small"
          onClick={() => onDecide(false)}
          disabled={disabled}
          loading={pendingAction === "reject"}
        >
          Cancel
        </Button>
        <Button
          variant="primary"
          size="small"
          onClick={() => onDecide(true)}
          disabled={disabled}
          loading={pendingAction === "approve"}
        >
          Keep going
        </Button>
      </div>
    </div>
  );
}
