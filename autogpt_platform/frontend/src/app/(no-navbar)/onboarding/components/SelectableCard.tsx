"use client";

import { cn } from "@/lib/utils";

interface Props {
  icon: React.ReactNode;
  label: string;
  selected: boolean;
  onClick: () => void;
}

export function SelectableCard({ icon, label, selected, onClick }: Props) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "flex flex-col items-center justify-center gap-2 rounded-xl border-2 bg-white p-4 transition-all hover:shadow-sm",
        selected
          ? "border-primary bg-primary/5 shadow-sm"
          : "border-transparent",
      )}
    >
      <span className="text-2xl">{icon}</span>
      <span className="text-sm font-medium">{label}</span>
    </button>
  );
}
