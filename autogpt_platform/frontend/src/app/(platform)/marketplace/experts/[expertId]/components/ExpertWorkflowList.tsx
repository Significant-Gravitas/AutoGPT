import { Expert } from "@/app/api/__generated__/models/expert";
import type { ExpertAccent } from "../../../components/ExpertsSection/helpers";
import { ExpertSection } from "./ExpertSection";
import { ExpertWorkflowCard } from "./ExpertWorkflowCard/ExpertWorkflowCard";

interface Props {
  name: string;
  workflows: Expert["workflows"];
  accent: ExpertAccent;
}

export function ExpertWorkflowList({ name, workflows, accent }: Props) {
  if (workflows.length === 0) return null;

  return (
    <ExpertSection
      title="Workflows"
      count={workflows.length}
      description={`Installed and ready to run the moment ${name} joins your team.`}
    >
      <ul className="grid gap-4 sm:grid-cols-2">
        {workflows.map((workflow) => (
          <ExpertWorkflowCard
            key={workflow.id}
            workflow={workflow}
            accent={accent}
          />
        ))}
      </ul>
    </ExpertSection>
  );
}
