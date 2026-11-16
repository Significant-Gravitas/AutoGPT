"use client";

import { SectionMobileNav } from "@/components/layout/SectionNav/SectionMobileNav";
import { useSettingsSidebar } from "../SettingsSidebar/useSettingsSidebar";

export function SettingsMobileNav() {
  const { items } = useSettingsSidebar();
  return <SectionMobileNav items={items} sectionName="Settings" />;
}
