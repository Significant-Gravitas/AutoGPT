import { useCallback, useEffect, useRef, useState } from "react";
import { ChevronDown, Check } from "lucide-react";
import { OnboardingStep } from "@/lib/autogpt-server-api";
import { useOnboarding } from "../../../../providers/onboarding/onboarding-provider";
import { cn } from "@/lib/utils";
import * as party from "party-js";

interface Task {
  id: OnboardingStep;
  name: string;
  amount: number;
  details: string;
  video?: string;
}

interface TaskGroup {
  name: string;
  tasks: Task[];
  isOpen: boolean;
}

export function TaskGroups() {
  const [groups, setGroups] = useState<TaskGroup[]>([
    {
      name: "Run your first agents",
      isOpen: true,
      tasks: [
        {
          id: "GET_RESULTS",
          name: "Complete onboarding and see your first agent's results",
          amount: 3,
          details: "",
        },
        {
          id: "RUN_AGENTS",
          name: "Run 10 agents",
          amount: 3,
          details: "Run agents from Library or Builder 10 times",
        },
      ],
    },
    {
      name: "Explore the Marketplace",
      isOpen: true,
      tasks: [
        {
          id: "MARKETPLACE_VISIT",
          name: "Go to Marketplace",
          amount: 0,
          details: "Click Marketplace in the top navigation",
          video: "/onboarding/marketplace-visit.mp4",
        },
        {
          id: "MARKETPLACE_ADD_AGENT",
          name: "Find an agent",
          amount: 1,
          details:
            "Search for an agent in the Marketplace, like a code generator or research assistant and add it to your Library",
          video: "/onboarding/marketplace-add.mp4",
        },
        {
          id: "MARKETPLACE_RUN_AGENT",
          name: "Try out your agent",
          amount: 1,
          details:
            "Run the agent you found in the Marketplace from the Library - whether it's a writing assistant, data analyzer, or something else",
          video: "/onboarding/marketplace-run.mp4",
        },
      ],
    },
    {
      name: "Build your own agent",
      isOpen: true,
      tasks: [
        {
          id: "BUILDER_OPEN",
          name: "Open the Builder",
          amount: 0,
          details: "Click Builder in the top navigation",
          video: "/onboarding/builder-open.mp4",
        },
        {
          id: "BUILDER_SAVE_AGENT",
          name: "Place your first blocks and save your agent",
          amount: 1,
          details:
            "Open block library on the left and add a block to the canvas then save your agent",
          video: "/onboarding/builder-save.mp4",
        },
        {
          id: "BUILDER_RUN_AGENT",
          name: "Run your agent",
          amount: 1,
          details: "Run your agent from the Builder",
          video: "/onboarding/builder-run.mp4",
        },
      ],
    },
  ]);
  const { state, updateState } = useOnboarding();
  const refs = useRef<Record<string, HTMLDivElement | null>>({});

  const toggleGroup = useCallback((name: string) => {
    setGroups((prevGroups) =>
      prevGroups.map((group) =>
        group.name === name ? { ...group, isOpen: !group.isOpen } : group,
      ),
    );
  }, []);

  const isTaskCompleted = useCallback(
    (task: Task) => {
      return state?.completedSteps?.includes(task.id) || false;
    },
    [state?.completedSteps],
  );

  const getCompletedCount = useCallback(
    (tasks: Task[]) => {
      return tasks.filter((task) => isTaskCompleted(task)).length;
    },
    [isTaskCompleted],
  );

  const isGroupCompleted = useCallback(
    (group: TaskGroup) => {
      return group.tasks.every((task) => isTaskCompleted(task));
    },
    [isTaskCompleted],
  );

  const setRef = (name: string) => (el: HTMLDivElement | null) => {
    if (el) {
      refs.current[name] = el;
    }
  };

  const delayConfetti = useCallback((el: HTMLDivElement, count: number) => {
    setTimeout(() => {
      party.confetti(el, {
        count,
        spread: 90,
        shapes: ["square", "circle"],
        size: party.variation.range(1, 1.5),
        speed: party.variation.range(250, 350),
        modules: [
          new party.ModuleBuilder()
            .drive("opacity")
            .by((t) => 1.4 - t)
            .through("lifetime")
            .build(),
        ],
      });
    }, 300);
  }, []);

  useEffect(() => {
    groups.forEach((group) => {
      const groupCompleted = isGroupCompleted(group);
      // Check if the last task in the group is completed
      const alreadyCelebrated = state?.notified.includes(
        group.tasks[group.tasks.length - 1].id,
      );

      if (groupCompleted) {
        const el = refs.current[group.name];
        if (el && !alreadyCelebrated) {
          delayConfetti(el, 50);
          // Update the state to include all group tasks as notified
          // This ensures that the confetti effect isn't perpetually triggered on Wallet
          const notifiedTasks = group.tasks.map((task) => task.id);
          updateState({
            notified: [...(state?.notified || []), ...notifiedTasks],
          });
        }
        return;
      }

      group.tasks.forEach((task) => {
        const el = refs.current[task.id];
        if (el && isTaskCompleted(task) && !state?.notified.includes(task.id)) {
          delayConfetti(el, 40);
          // Update the state to include the task as notified
          updateState({ notified: [...(state?.notified || []), task.id] });
        }
      });
    });
  }, [
    state?.completedSteps,
    delayConfetti,
    groups,
    updateState,
    state?.notified,
    isGroupCompleted,
    isTaskCompleted,
  ]);

  return (
    <div className="space-y-2">
      {groups.map((group) => (
        <div
          key={group.name}
          ref={setRef(group.name)}
          className="mt-3 overflow-hidden rounded-lg border border-zinc-200 bg-zinc-100"
        >
          {/* Group Header */}
          <div
            className="flex cursor-pointer items-center justify-between p-3"
            onClick={() => toggleGroup(group.name)}
          >
            {/* Name and completed count */}
            <div className="flex-1">
              <div className="text-sm font-medium text-zinc-900">
                {group.name}
              </div>
              <div className="mt-1 text-xs font-normal leading-tight text-zinc-500">
                {getCompletedCount(group.tasks)} of {group.tasks.length}{" "}
                completed
              </div>
            </div>
            {/* Reward and chevron */}
            <div className="flex items-center gap-2">
              {isGroupCompleted(group) ? (
                <div className="rounded-full bg-green-200 px-2.5 py-0.5 font-sans text-xs font-medium text-green-700">
                  Done
                </div>
              ) : (
                <div className="text-xs font-medium leading-tight text-violet-600">
                  $
                  {group.tasks
                    .reduce((sum, task) => sum + task.amount, 0)
                    .toFixed(2)}
                </div>
              )}
              <ChevronDown
                className={`h-5 w-5 text-slate-950 transition-transform duration-300 ease-in-out ${group.isOpen ? "rotate-180" : ""}`}
              />
            </div>
          </div>

          {/* Tasks */}
          <div
            className={cn(
              "overflow-hidden transition-all duration-300 ease-in-out",
              group.isOpen || !isGroupCompleted(group)
                ? "max-h-[1000px] opacity-100"
                : "max-h-0 opacity-0",
            )}
          >
            {group.tasks.map((task) => (
              <div
                key={task.id}
                ref={setRef(task.id)}
                className="mx-3 border-t border-zinc-300 px-1 pb-1 pt-3"
              >
                <div className="mb-2 flex items-center justify-between">
                  {/* Checkmark and name */}
                  <div className="flex items-center gap-2">
                    <div
                      className={cn(
                        "flex h-4 w-4 items-center justify-center rounded-full border",
                        isTaskCompleted(task)
                          ? "border-emerald-600"
                          : "border-zinc-600",
                      )}
                    >
                      {isTaskCompleted(task) && (
                        <Check className="h-3 w-3 text-emerald-600" />
                      )}
                    </div>
                    <span
                      className={cn(
                        "text-sm font-normal",
                        isTaskCompleted(task)
                          ? "text-zinc-500 line-through"
                          : "text-zinc-800",
                      )}
                    >
                      {task.name}
                    </span>
                  </div>
                  {/* Reward */}
                  {task.amount > 0 && (
                    <span
                      className={cn(
                        "text-xs font-normal text-zinc-500",
                        isTaskCompleted(task) ? "line-through" : "",
                      )}
                    >
                      ${task.amount.toFixed(2)}
                    </span>
                  )}
                </div>

                {/* Details section */}
                {!isGroupCompleted(group) && (
                  <>
                    <div
                      className={cn(
                        "overflow-hidden pl-6 text-xs font-normal text-zinc-500 transition-all duration-300 ease-in-out",
                        isTaskCompleted(task) && "line-through",
                        group.isOpen
                          ? "max-h-[100px] opacity-100"
                          : "max-h-0 opacity-0",
                      )}
                    >
                      {task.details}
                    </div>
                    {task.video ? (
                      <div
                        className={cn(
                          "relative mx-6 aspect-video overflow-hidden rounded-lg transition-all duration-300 ease-in-out",
                          group.isOpen
                            ? "my-2 max-h-[200px] opacity-100"
                            : "max-h-0 opacity-0",
                        )}
                      >
                        <video
                          src={task.video}
                          autoPlay
                          loop
                          muted
                          playsInline
                          className={cn(
                            "h-full w-full object-cover object-center",
                            isTaskCompleted(task) && "grayscale",
                          )}
                        ></video>
                      </div>
                    ) : (
                      <div className="mb-1" />
                    )}
                  </>
                )}
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}
