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
      className="flex items-center gap-1 rounded-full bg-zinc-100 p-0.5"
    >
      {tabs.map((tab) => (
        <button
          key={tab.id}
          role="tab"
          type="button"
          aria-selected={activeTab === tab.id}
          onClick={() => onChange(tab.id)}
          className={cn(
            "flex-1 rounded-full px-2.5 py-0.5 text-xs font-medium transition-colors",
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
