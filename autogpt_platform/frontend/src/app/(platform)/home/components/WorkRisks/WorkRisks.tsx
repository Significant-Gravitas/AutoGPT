import Link from "next/link";
import type { HomeDashboardResponse } from "@/app/api/__generated__/models/homeDashboardResponse";
import { Text } from "@/components/atoms/Text/Text";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import { founderSafeText } from "@/lib/founder-safe-text";
import { HomeTile } from "../HomeTile/HomeTile";

interface Props {
  dashboard: HomeDashboardResponse;
}

export function WorkRisks({ dashboard }: Props) {
  const workRisks = (dashboard.work_items ?? []).filter((item) =>
    ["partial", "blocked_manager", "failed"].includes(item.status),
  );
  const runRisks = dashboard.briefing.outcomes.filter(
    (outcome) => outcome.status === "failed" || outcome.status === "partial",
  );
  const riskCount = workRisks.length + runRisks.length;

  return (
    <HomeTile
      title={<Text variant="h5">Risks</Text>}
      header={
        <Text variant="large" className="text-zinc-600">
          Problems AutoPilot is resolving or may need to escalate.
        </Text>
      }
      contentClassName="flex flex-col"
    >
      {riskCount === 0 ? (
        <div className="py-6 text-center">
          <Text variant="body-medium" className="text-zinc-800">
            No known risks
          </Text>
          <Text variant="small" className="mt-1 text-zinc-500">
            Your team can keep moving.
          </Text>
        </div>
      ) : (
        <div className="divide-y divide-zinc-100">
          {workRisks.map((item) => (
            <RiskRow
              key={item.id}
              href={item.link}
              title={founderSafeText(item.title, "Expert work needs attention")}
              description={founderSafeText(
                item.blocker || item.expected_deliverable,
                "AutoPilot is reviewing the next safe step.",
              )}
              expertName={item.expert.name}
              expertAvatar={item.expert.avatar_url}
              label={
                item.status === "blocked_manager"
                  ? "Needs AutoPilot"
                  : item.status === "partial"
                    ? "Partial"
                    : "Failed"
              }
              confidence={confidenceLabel(item.confidence)}
            />
          ))}
          {runRisks.map((outcome) => (
            <RiskRow
              key={outcome.id}
              href={outcome.link}
              title={outcome.title}
              description={outcome.summary}
              expertName={outcome.expert?.name ?? outcome.agent_name}
              expertAvatar={outcome.expert?.avatar_url}
              label={outcome.status === "partial" ? "Needs attention" : "Failed"}
            />
          ))}
        </div>
      )}
    </HomeTile>
  );
}

interface RowProps {
  href?: string | null;
  title: string;
  description: string;
  expertName: string;
  expertAvatar?: string | null;
  label: string;
  confidence?: string;
}

function RiskRow({
  href,
  title,
  description,
  expertName,
  expertAvatar,
  label,
  confidence,
}: RowProps) {
  const content = (
    <>
      <ExpertAvatar
        name={expertName}
        avatarUrl={expertAvatar ?? null}
        size={32}
      />
      <div className="min-w-0 flex-1">
        <Text variant="body-medium" className="line-clamp-1 text-zinc-900">
          {title}
        </Text>
        <Text variant="small" className="mt-0.5 line-clamp-2 text-zinc-500">
          {description}
        </Text>
        <div className="mt-1 flex items-center gap-1.5 text-xs text-zinc-400">
          <span>{label}</span>
          {confidence ? (
            <>
              <span aria-hidden="true">·</span>
              <span>{confidence}</span>
            </>
          ) : null}
        </div>
      </div>
    </>
  );
  const className = "-mx-2 flex gap-3 rounded-xl px-2 py-3";
  if (!href) return <div className={className}>{content}</div>;
  return (
    <Link
      href={href}
      className={`${className} transition-colors hover:bg-zinc-50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-zinc-400`}
    >
      {content}
    </Link>
  );
}

function confidenceLabel(value: string) {
  return value.charAt(0).toUpperCase() + value.slice(1);
}
