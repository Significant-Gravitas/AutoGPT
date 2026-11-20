"use client";

import { SectionSidebar } from "@/components/layout/SectionNav/SectionSidebar";
import { useSettingsSidebar } from "./useSettingsSidebar";

export function SettingsSidebar() {
  const { items } = useSettingsSidebar();
  return <SectionSidebar items={items} className="w-[237px]" />;
}
