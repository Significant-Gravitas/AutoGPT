"use client";

import Link from "next/link";
import { ExpertIntegrationsSection } from "@/app/(platform)/team/[expertId]/components/ExpertIntegrationsSection/ExpertIntegrationsSection";
import { useCopilotUIStore } from "../../../../store";

/** The chat's expert's integrations, with the same add / use existing /
 *  remove actions as the Integrations tab on the expert's page. */
export function IntegrationsTab() {
  const expert = useCopilotUIStore((s) => s.contextPanelExpert);
  if (!expert) return null;

  return (
    <div className="flex flex-col gap-3">
      <ExpertIntegrationsSection
        expertId={expert.id}
        expertName={expert.name}
        compact
      />
      <Link
        href={`/team/${expert.id}`}
        className="px-1 text-xs text-zinc-500 underline hover:text-zinc-800"
      >
        Open {expert.name}&apos;s page
      </Link>
    </div>
  );
}
