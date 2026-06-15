import { cn } from "@/lib/utils";
import type { ContextPanelTab } from "../../../store";

interface Props {
  activeTab: ContextPanelTab;
  filesCount: number;
  onChange: (tab: ContextPanelTab) => void;
}

export function TabSwitcher({ activeTab, filesCount, onChange }: Props) {
  const tabs: { id: ContextPanelTab; label: string }[] = [
    { id: "progress", label: "Progress" },
    { id: "files", label: `Files (${filesCount})` },
  ];

  return (
    <div
      role="tablist"
      aria-label="Context panel sections"
      className="flex items-center rounded-md border border-zinc-200 bg-zinc-50 p-0.5 text-xs font-medium"
    >
      {tabs.map((tab) => (
        <button
          key={tab.id}
          role="tab"
          type="button"
          aria-selected={activeTab === tab.id}
          onClick={() => onChange(tab.id)}
          className={cn(
            "rounded px-2 py-1 transition-colors",
            activeTab === tab.id
              ? "bg-white text-zinc-900 shadow-sm"
              : "text-zinc-500 hover:text-zinc-700",
          )}
        >
          {tab.label}
        </button>
      ))}
    </div>
  );
}
