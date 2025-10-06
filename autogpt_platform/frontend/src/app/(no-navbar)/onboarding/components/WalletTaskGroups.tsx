import { useCallback, useEffect, useRef, useState } from "react";
import { ChevronDown, Check, BadgeQuestionMark } from "lucide-react";
import { cn } from "@/lib/utils";
import * as party from "party-js";
import { useOnboarding } from "@/providers/onboarding/onboarding-provider";
import { Task, TaskGroup } from "@/components/__legacy__/Wallet";

interface Props {
  groups: TaskGroup[];
}

export function TaskGroups({ groups }: Props) {
  const { state, updateState } = useOnboarding();
  const refs = useRef<Record<string, HTMLDivElement | null>>({});

  const [openGroups, setOpenGroups] = useState<Record<string, boolean>>(() => {
    const initialState: Record<string, boolean> = {};
    groups.forEach((group) => {
      initialState[group.name] = true;
    });
    return initialState;
  });

  const toggleGroup = useCallback((name: string) => {
    setOpenGroups((prev) => ({
      ...prev,
      [name]: !prev[name],
    }));
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

  useEffect(() => {
    // Close completed groups
    setOpenGroups((prevGroups) =>
      groups.reduce(
        (acc, group) => {
          acc[group.name] = isGroupCompleted(group)
            ? false
            : prevGroups[group.name];
          return acc;
        },
        {} as Record<string, boolean>,
      ),
    );
  }, [state?.completedSteps, isGroupCompleted]);

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
          className="mt-3 overflow-hidden rounded-lg border border-zinc-100 bg-zinc-50"
        >
          {/* Group Header */}
          <div
            className="flex cursor-pointer items-center justify-between p-3"
            onClick={() => toggleGroup(group.name)}
          >
            {/* Name, details and completed count */}
            <div className="flex-1">
              <div className="text-sm font-medium text-zinc-900">
                {group.name}
              </div>
              <div className="mt-1 text-xs font-normal leading-tight text-zinc-500">
                {group.details}
                <br />
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
                className={`h-5 w-5 text-slate-950 transition-transform duration-300 ease-in-out ${openGroups[group.name] ? "rotate-180" : ""}`}
              />
            </div>
          </div>

          {/* Tasks */}
          <div
            className={cn(
              "overflow-hidden transition-all duration-300 ease-in-out",
              openGroups[group.name] || !isGroupCompleted(group)
                ? "max-h-[1200px] opacity-100"
                : "max-h-0 opacity-0",
            )}
          >
            {group.tasks.map((task) => (
              <div
                key={task.id}
                ref={setRef(task.id)}
                className="mx-3 border-t border-zinc-200 px-1 pb-0.5 pt-3"
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
                {/* Progress bar and counter text */}
                {task.progress && !isTaskCompleted(task) && (
                  <div className="mb-1 flex w-full items-center justify-between pl-6 pr-3">
                    <div className="h-2 flex-1 overflow-hidden rounded-full bg-zinc-100">
                      <div
                        className="h-full bg-violet-400 transition-all duration-500 ease-in-out"
                        style={{
                          width: `${Math.min(
                            100,
                            (task.progress.current / task.progress.target) *
                              100,
                          )}%`,
                        }}
                      />
                    </div>
                    <span className="mx-1 w-8 text-right text-xs font-normal text-zinc-500">
                      {(
                        (task.progress.current / task.progress.target) *
                        100
                      ).toFixed(0)}
                      %
                    </span>
                  </div>
                )}
                {/* Details section */}
                {!isGroupCompleted(group) && (
                  <>
                    <div
                      className={cn(
                        "mt-0 overflow-hidden pl-6 pt-0 text-xs font-normal text-zinc-500 transition-all duration-300 ease-in-out",
                        isTaskCompleted(task) && "line-through",
                        openGroups[group.name]
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
                          openGroups[group.name]
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
      {/* Hidden Tasks group */}
      <div className="mt-3 overflow-hidden rounded-lg border border-zinc-100 bg-zinc-50">
        {/* Group Header */}
        <div className="flex items-center justify-between p-3">
          {/* Name and details */}
          <div className="flex-1">
            <div className="text-sm font-medium text-zinc-900">
              Hidden Tasks
            </div>
            <div className="mt-1 text-xs font-normal leading-tight text-zinc-500">
              Check back later — new tasks are on the way
            </div>
          </div>
        </div>
        {/* Tasks */}
        <div>
          <div className="mx-3 border-t border-zinc-200 px-1 pb-1 pt-3">
            <div className="mb-2 flex items-center justify-between">
              {/* Question mark and rectangle */}
              <div className="flex items-center gap-2">
                <div className="flex h-4 w-4 items-center justify-center">
                  <BadgeQuestionMark />
                </div>
                <div className="h-4 w-64 rounded-full bg-zinc-100" />
              </div>
            </div>
          </div>
          <div className="mx-3 border-t border-zinc-200 px-1 pb-1 pt-3">
            <div className="mb-2 flex items-center justify-between">
              {/* Question mark and rectangle */}
              <div className="flex items-center gap-2">
                <div className="flex h-4 w-4 items-center justify-center">
                  <BadgeQuestionMark />
                </div>
                <div className="h-4 w-64 rounded-full bg-zinc-100" />
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
