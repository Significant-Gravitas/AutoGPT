"use client";

import { SegmentedControl } from "@/components/molecules/SegmentedControl/SegmentedControl";
import type { HomeView } from "./useHomeView";

const OPTIONS = [
  { value: "chat" as const, label: "Chat" },
  { value: "board" as const, label: "Board" },
];

interface Props {
  value: HomeView;
  onChange: (view: HomeView) => void;
}

export function HomeViewTabs(props: Props) {
  return (
    <div className="pointer-events-none relative z-30 flex shrink-0 justify-center pt-4 md:pt-6 [&>*]:pointer-events-auto">
      <HomeViewSwitch {...props} />
    </div>
  );
}

export function HomeViewSwitch({ value, onChange }: Props) {
  return (
    <SegmentedControl
      value={value}
      onChange={onChange}
      options={OPTIONS}
      ariaLabel="Home view"
    />
  );
}
