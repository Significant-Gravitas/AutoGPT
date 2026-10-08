import { scenarios, type Scenario } from "@/lib/openui/scenarios";
import { Icon } from "@/components/atoms/Icon/Icon";
import {
  AnalyticsUpIcon,
  UserSearch01Icon,
  Rocket01Icon,
  ArrowRight01Icon,
} from "@hugeicons/core-free-icons";
import { cn } from "@/lib/utils";

const icons = [AnalyticsUpIcon, UserSearch01Icon, Rocket01Icon];

interface Props {
  selected: Scenario;
  onSelect: (scenario: Scenario) => void;
}

export function ScenarioPicker({ selected, onSelect }: Props) {
  return (
    <div
      className="flex shrink-0 gap-2 overflow-x-auto px-5 pb-4 sm:grid sm:grid-cols-3 sm:px-8"
      aria-label="Example workflows"
    >
      {scenarios.map((scenario, index) => (
        <button
          key={scenario.id}
          aria-pressed={selected.id === scenario.id}
          onClick={() => onSelect(scenario)}
          className={cn(
            "flex shrink-0 items-center gap-3 rounded-xl border border-zinc-200 bg-white px-4 py-3 text-left transition-colors hover:border-purple-200 focus-visible:outline-purple-500",
            selected.id === scenario.id && "border-purple-200 bg-purple-50/40",
          )}
        >
          <span
            className={cn(
              "flex size-9 shrink-0 items-center justify-center rounded-lg bg-zinc-50 text-zinc-500",
              selected.id === scenario.id && "bg-purple-100 text-purple-600",
            )}
          >
            <Icon icon={icons[index]} size={18} />
          </span>
          <span className="min-w-0">
            <span className="block text-xs font-semibold text-zinc-800">
              {scenario.name}
            </span>
            <span className="mt-0.5 hidden text-[11px] text-zinc-500 xl:block">
              {scenario.label}
            </span>
          </span>
          <Icon
            icon={ArrowRight01Icon}
            size={14}
            className={cn(
              "ml-auto shrink-0 text-zinc-300",
              selected.id === scenario.id && "text-purple-500",
            )}
          />
        </button>
      ))}
    </div>
  );
}
