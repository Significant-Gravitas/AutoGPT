"use client";

import { Tabs, TabsList, TabsTrigger } from "@/components/__legacy__/ui/tabs";

export type BuilderView = "old" | "new";

export function BuilderViewTabs({
  value,
  onChange,
}: {
  value: BuilderView;
  onChange: (value: BuilderView) => void;
}) {
  return (
    <div className="pointer-events-auto fixed right-4 top-20 z-50">
      <Tabs
        value={value}
        onValueChange={(v: string) => onChange(v as BuilderView)}
      >
        <TabsList className="w-fit bg-zinc-900">
          <TabsTrigger value="old" className="text-gray-100">
            Old
          </TabsTrigger>
          <TabsTrigger value="new" className="text-gray-100">
            New
          </TabsTrigger>
        </TabsList>
      </Tabs>
    </div>
  );
}
