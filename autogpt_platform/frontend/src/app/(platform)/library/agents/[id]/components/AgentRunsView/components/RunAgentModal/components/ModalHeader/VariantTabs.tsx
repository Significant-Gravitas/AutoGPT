import { cn } from "@/lib/utils";
import { RunVariant } from "../../useAgentRunModal";

interface VariantTabsProps {
  activeVariant: RunVariant;
  onVariantChange: (variant: RunVariant) => void;
  hasExternalTrigger: boolean;
}

interface TabConfig {
  variant: RunVariant;
  label: string;
  description: string;
  available: boolean;
}

export function VariantTabs({
  activeVariant,
  onVariantChange,
  hasExternalTrigger,
}: VariantTabsProps) {
  const tabs: TabConfig[] = [
    {
      variant: "manual",
      label: "Manual Run",
      description: "Run the agent once with custom inputs",
      available: !hasExternalTrigger,
    },
    {
      variant: "schedule",
      label: "Schedule Run",
      description: "Run the agent on a recurring schedule",
      available: !hasExternalTrigger,
    },
    {
      variant: "automatic-trigger",
      label: "Automatic Trigger",
      description: "Set up webhook-based automatic execution",
      available: hasExternalTrigger,
    },
    {
      variant: "manual-trigger",
      label: "Manual Trigger",
      description: "Trigger the agent manually via API",
      available: hasExternalTrigger,
    },
  ];

  const availableTabs = tabs.filter((tab) => tab.available);

  return (
    <div className="border-b border-neutral-100">
      <nav className="flex space-x-8">
        {availableTabs.map((tab) => (
          <button
            key={tab.variant}
            onClick={() => onVariantChange(tab.variant)}
            className={cn(
              "border-b-2 px-1 py-2 text-sm font-medium transition-colors",
              activeVariant === tab.variant
                ? "border-blue-500 text-blue-600"
                : "border-transparent text-neutral-500 hover:border-neutral-300 hover:text-neutral-700",
            )}
          >
            <div className="text-left">
              <div className="font-medium">{tab.label}</div>
              <div className="mt-0.5 text-xs text-neutral-400">
                {tab.description}
              </div>
            </div>
          </button>
        ))}
      </nav>
    </div>
  );
}
