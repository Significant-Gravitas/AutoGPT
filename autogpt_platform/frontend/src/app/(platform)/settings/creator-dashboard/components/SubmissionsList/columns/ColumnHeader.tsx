import type { ReactNode } from "react";

interface Props {
  label: string;
  filter?: ReactNode;
  align?: "left" | "right";
  width?: string;
}

export function ColumnHeader({ label, filter, align = "left", width }: Props) {
  return (
    <th
      scope="col"
      style={width ? { width } : undefined}
      className={`px-4 py-3 ${align === "right" ? "text-right" : "text-left"}`}
    >
      <div
        className={`inline-flex items-center gap-1.5 text-xs font-medium uppercase tracking-wide text-zinc-500 ${
          align === "right" ? "flex-row-reverse" : ""
        }`}
      >
        <span>{label}</span>
        {filter}
      </div>
    </th>
  );
}
