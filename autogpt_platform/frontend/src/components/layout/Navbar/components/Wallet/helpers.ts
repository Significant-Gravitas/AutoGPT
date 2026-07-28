import { OnboardingStep, UserOnboarding } from "@/lib/autogpt-server-api";

export interface Task {
  id: OnboardingStep;
  name: string;
  amount: number;
  details: string;
  video?: string;
  progress?: {
    current: number;
    target: number;
  };
}

export interface TaskGroup {
  name: string;
  details: string;
  tasks: Task[];
}

export function getTaskGroups(state: UserOnboarding | null): TaskGroup[] {
  return [
    {
      name: "First Wins",
      details: "Kickstart your journey with quick wins.",
      tasks: [
        {
          id: "VISIT_COPILOT",
          name: "Complete onboarding",
          amount: 3,
          details: "",
        },
        {
          id: "MARKETPLACE_ADD_AGENT",
          name: "Get an agent from the marketplace",
          amount: 1,
          details:
            "Search for an agent in the Marketplace and add it to your Library",
          video: "/onboarding/marketplace-add.mp4",
        },
        {
          id: "MARKETPLACE_RUN_AGENT",
          name: "Open the Library page and run an agent",
          amount: 1,
          details: "Go to the Library, open an agent you want, and run it",
          video: "/onboarding/agent-run.mp4",
        },
      ],
    },
    {
      name: "Consistency Challenge",
      details: "Build your rhythm and make agents part of your routine.",
      tasks: [
        {
          id: "SCHEDULE_AGENT",
          name: "Schedule your first agent",
          amount: 1,
          details: "Schedule an agent to run on a recurring basis",
          video: "/onboarding/agent-schedule.mp4",
        },
        {
          id: "RUN_3_DAYS",
          name: "Run agents 3 days in a row",
          amount: 1,
          details:
            "Run any agents from the Library or Builder for 3 days in a row",
          progress: {
            current: state?.consecutiveRunDays || 0,
            target: 3,
          },
        },
      ],
    },
    {
      name: "The Pro Playground",
      details: "Master powerful features to supercharge your workflow.",
      tasks: [
        {
          id: "TRIGGER_WEBHOOK",
          name: "Trigger an agent via webhook",
          amount: 1,
          details:
            "In the Builder, go to Settings and copy the Webhook URL. Use it to trigger your agent from another app.",
        },
        {
          id: "RUN_14_DAYS",
          name: "Run agents 14 days in a row",
          amount: 1,
          details:
            "Run any agents from the Library or Builder for 14 days in a row",
          progress: {
            current: state?.consecutiveRunDays || 0,
            target: 14,
          },
        },
        {
          id: "RUN_AGENTS_100",
          name: "Complete 100 agent runs",
          amount: 1,
          details: "Let your agents run and complete 100 tasks in total",
          progress: {
            current: state?.agentRuns || 0,
            target: 100,
          },
        },
      ],
    },
  ];
}

export interface EarnRow {
  key: string;
  label: string;
  done: boolean;
  amount: number;
}

/**
 * Flattens the task groups into a single list for the compact panel: a fully
 * completed group collapses to one "Done" row, while any group still in
 * progress contributes one row per task.
 */
export function getEarnRows(
  groups: TaskGroup[],
  // The onboarding payload is typed as always carrying `completedSteps`, but
  // the rest of the wallet already guards against it being absent — do the same
  // here so a thin backend response degrades to "nothing claimed" and not a
  // TypeError.
  completedSteps: OnboardingStep[] | undefined,
): EarnRow[] {
  const claimed = completedSteps ?? [];

  return groups.flatMap((group): EarnRow[] => {
    const completed = group.tasks.filter((task) => claimed.includes(task.id));

    if (completed.length === group.tasks.length) {
      return [
        {
          key: group.name,
          label: `${group.name} · ${completed.length} of ${group.tasks.length}`,
          done: true,
          amount: 0,
        },
      ];
    }

    return group.tasks.map((task) => ({
      key: task.id,
      label: task.name,
      done: claimed.includes(task.id),
      amount: task.amount,
    }));
  });
}
