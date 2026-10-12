import { ExpertWorkflowRef } from "@/app/api/__generated__/models/expertWorkflowRef";
import { GraphExecutionMeta } from "@/app/api/__generated__/models/graphExecutionMeta";
import { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import {
  Alert01Icon,
  Calendar03Icon,
  FlashIcon,
  PlayIcon,
} from "@hugeicons/core-free-icons";
import { useRouter } from "next/navigation";
import { workflowNeedsSetup } from "../../helpers";
import { getWorkflowCredentialProviders } from "./WorkflowCredentialStack";

const STATUS = {
  "needs-setup": {
    label: "Needs setup",
    className: "bg-amber-50 text-amber-700",
    icon: Alert01Icon,
    iconClassName: "bg-amber-50 text-amber-600",
  },
  scheduled: {
    label: "Scheduled",
    className: "bg-white text-zinc-700",
    icon: Calendar03Icon,
    iconClassName: "bg-violet-50 text-violet-600",
  },
  trigger: {
    label: "Triggered",
    className: "bg-white text-zinc-700",
    icon: FlashIcon,
    iconClassName: "bg-sky-50 text-sky-600",
  },
  manual: {
    label: "Manual",
    className: "bg-white text-zinc-500",
    icon: PlayIcon,
    iconClassName: "bg-zinc-100 text-zinc-500",
  },
} as const;

interface Args {
  workflow: ExpertWorkflowRef;
  libraryAgent: LibraryAgent | undefined;
  expertId?: string;
}

export function useExpertWorkflowCard({
  workflow,
  libraryAgent,
  expertId,
}: Args) {
  const router = useRouter();
  const name = workflow.name ?? "Unnamed workflow";
  const needsSetup = workflowNeedsSetup(workflow);
  const libraryHref = workflow.library_agent_id
    ? `/library/agents/${workflow.library_agent_id}`
    : null;
  const chatPrompt = `Tell me about the workflow "${name}" and how you use it.`;

  function openRun(execution: GraphExecutionMeta) {
    router.push(`${libraryHref}?activeTab=runs&activeItem=${execution.id}`);
  }

  function openTriggers() {
    router.push(`${libraryHref}?activeTab=triggers`);
  }

  const isTriggerWorkflow =
    Boolean(libraryAgent?.trigger_setup_info) ||
    (workflow.chain ?? []).some((item) => item.kind === "trigger");
  const statusKind = needsSetup
    ? "needs-setup"
    : workflow.schedule_cron
      ? "scheduled"
      : isTriggerWorkflow
        ? "trigger"
        : "manual";

  return {
    name,
    needsSetup,
    runCount: libraryAgent?.execution_count,
    isTriggerWorkflow,
    status: STATUS[statusKind],
    credentialProviders: getWorkflowCredentialProviders(
      libraryAgent,
      workflow.chain ?? [],
    ),
    libraryHref,
    builderHref: workflow.graph_id
      ? `/build?flowID=${workflow.graph_id}`
      : null,
    chatPrompt,
    chatHref: `/home?${expertId ? `expertId=${expertId}&` : ""}autosubmit=true#prompt=${encodeURIComponent(chatPrompt)}`,
    openRun,
    openTriggers,
  };
}
