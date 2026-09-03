import { Expert } from "@/app/api/__generated__/models/expert";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { cn } from "@/lib/utils";
import {
  CheckmarkCircle02Icon,
  FlashIcon,
  SparklesIcon,
} from "@hugeicons/core-free-icons";
import type { ExpertAccent } from "../../../components/ExpertsSection/helpers";

interface Props {
  expert: Expert;
  accent: ExpertAccent;
  hiredExpert: Expert | null;
  isHiring: boolean;
  onHire: () => void;
}

/** The page's one call to action, pinned beside the profile. */
export function ExpertHireCard({
  expert,
  accent,
  hiredExpert,
  isHiring,
  onHire,
}: Props) {
  const workflowCount = expert.workflows.length;
  const skillCount = expert.skills?.length ?? 0;

  return (
    <aside className="flex flex-col gap-5 rounded-3xl border border-zinc-200/80 bg-white p-6 shadow-[0_1px_2px_rgba(16,24,40,0.04)] lg:sticky lg:top-24">
      <dl className="flex flex-col gap-3 text-sm">
        <div className="flex items-center gap-2.5">
          <Icon icon={FlashIcon} size={16} className={accent.icon} />
          <dt className="sr-only">Preloaded workflows</dt>
          <dd className="text-zinc-700">
            {workflowCount}{" "}
            {workflowCount === 1
              ? "workflow ready on day one"
              : "workflows ready on day one"}
          </dd>
        </div>
        {skillCount > 0 ? (
          <div className="flex items-center gap-2.5">
            <Icon icon={SparklesIcon} size={16} className={accent.icon} />
            <dt className="sr-only">Skills</dt>
            <dd className="text-zinc-700">
              {skillCount} {skillCount === 1 ? "skill" : "skills"}
            </dd>
          </div>
        ) : null}
      </dl>

      {hiredExpert ? (
        <>
          <div className="flex h-12 items-center justify-center gap-2 rounded-full border border-emerald-200 bg-emerald-50 text-base font-medium text-emerald-700">
            <Icon icon={CheckmarkCircle02Icon} size={20} />
            On your team
          </div>
          <Button
            as="NextLink"
            href={`/copilot?expertId=${hiredExpert.id}`}
            variant="secondary"
            size="large"
            className="w-full rounded-full"
          >
            {`Chat with ${expert.name}`}
          </Button>
        </>
      ) : (
        <>
          <Button
            variant="primary"
            onClick={onHire}
            loading={isHiring}
            className={cn("h-12 w-full rounded-full text-base")}
          >
            {`Hire ${expert.name}`}
          </Button>
          <p className="text-center text-xs text-zinc-500">
            {expert.name} joins your team with{" "}
            {workflowCount === 1 ? "this workflow" : "these workflows"}{" "}
            preloaded, ready to start in minutes.
          </p>
        </>
      )}
    </aside>
  );
}
