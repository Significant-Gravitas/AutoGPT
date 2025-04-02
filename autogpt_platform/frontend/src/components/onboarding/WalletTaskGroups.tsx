import { useState } from "react";
import { ChevronDown, Check } from "lucide-react";
import { OnboardingStep } from "@/lib/autogpt-server-api";
import { useOnboarding } from "./onboarding-provider";
import { cn } from "@/lib/utils";
import Image from "next/image";

interface Task {
  id: OnboardingStep;
  name: string;
  amount: number;
  details: string;
  gif?: string;
}

interface TaskGroup {
  name: string;
  tasks: Task[];
  isOpen: boolean;
}

export function TaskGroups() {
  const [groups, setGroups] = useState<TaskGroup[]>([
    {
      name: "Run your first agent",
      isOpen: false,
      tasks: [
        {
          id: "CONGRATS",
          name: "Finish onboarding",
          amount: 3,
          details: "Go through our step by step tutorial",
        },
        {
          id: "GET_RESULTS",
          name: "Get results from first agent",
          amount: 3,
          details:
            "Sit back and relax - your agent is running and will finish soon! See the results in the Library once it's done",
        },
      ],
    },
    {
      name: "Explore the Marketplace",
      isOpen: false,
      tasks: [
        {
          id: "MARKETPLACE_VISIT",
          name: "Go to Marketplace",
          amount: 0,
          details: "Click Marketplace in the top navigation",
        },
        {
          id: "MARKETPLACE_ADD_AGENT",
          name: "Find an agent",
          amount: 1,
          details:
            "Search for an agent in the Marketplace, like a code generator or research assistant and add it to your Library",
        },
        {
          id: "MARKETPLACE_RUN_AGENT",
          name: "Try out your agent",
          amount: 1,
          details:
            "Run the agent you found in the Marketplace from the Library - whether it's a writing assistant, data analyzer, or something else",
        },
      ],
    },
    {
      name: "Build your own agent",
      isOpen: false,
      tasks: [
        {
          id: "BUILDER_OPEN",
          name: "Open the Builder",
          amount: 0,
          details: "Click Builder in the top navigation",
        },
        {
          id: "BUILDER_SAVE_AGENT",
          name: "Place your first blocks and save your agent",
          amount: 1,
          details:
            "Open block library on the left and add a block to the canvas then save your agent",
        },
        {
          id: "BUILDER_RUN_AGENT",
          name: "Run your agent",
          amount: 1,
          details: "Run your agent from the Builder",
        },
      ],
    },
  ]);

  const { state } = useOnboarding();

  const toggleGroup = (name: string) => {
    setGroups(
      groups.map((group) =>
        group.name === name ? { ...group, isOpen: !group.isOpen } : group,
      ),
    );
  };

  const isTaskCompleted = (task: Task) => {
    return state?.completedSteps?.includes(task.id) || false;
  };

  const getCompletedCount = (tasks: Task[]) => {
    return tasks.filter((task) => isTaskCompleted(task)).length;
  };

  return (
    <div className="space-y-2">
      {groups.map((group) => (
        <div key={group.name} className="mt-2 overflow-hidden rounded-lg">
          {/* Group Header - unchanged */}
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
              <div className="text-xs font-medium leading-tight text-violet-600">
                $
                {group.tasks
                  .reduce((sum, task) => sum + task.amount, 0)
                  .toFixed(2)}
              </div>
              <ChevronDown
                className={`h-5 w-5 text-slate-950 transition-transform duration-300 ease-in-out ${
                  group.isOpen ? "rotate-180" : ""
                }`}
              />
            </div>
          </div>

          {/* Always visible tasks */}
          <div className="">
            {group.tasks.map((task) => (
              <div key={task.id} className="mt-1 px-4 py-1">
                <div className="flex items-center justify-between">
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
                  <span
                    className={cn(
                      "text-xs font-normal text-zinc-500",
                      isTaskCompleted(task) ? "line-through" : "",
                    )}
                  >
                    ${task.amount.toFixed(2)}
                  </span>
                </div>

                {/* Details section */}
                <div
                  className={cn(
                    "mt-2 overflow-hidden pl-6 text-xs font-normal text-zinc-500 transition-all duration-300 ease-in-out",
                    isTaskCompleted(task) && "line-through",
                    group.isOpen
                      ? "max-h-[100px] opacity-100"
                      : "max-h-0 opacity-0",
                  )}
                >
                  {task.details}
                </div>
                {true && (
                  <div
                    className={cn(
                      "relative mx-6 aspect-video overflow-hidden rounded-lg transition-all duration-300 ease-in-out",
                      group.isOpen
                        ? "my-2 max-h-[200px] opacity-100"
                        : "max-h-0 opacity-0",
                    )}
                  >
                    <Image
                      src={task.gif || "/rat-spinning.gif"}
                      alt="GIF task instructions"
                      fill
                      className={cn("object-cover object-center")}
                    />
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}
