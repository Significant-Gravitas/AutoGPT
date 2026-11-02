"use client";

import { SectionSidebar } from "@/components/layout/SectionNav/SectionSidebar";
import { useAdminSidebar } from "./useAdminSidebar";

export function AdminSidebar() {
  const { items } = useAdminSidebar();
  return <SectionSidebar items={items} className="w-[268px]" />;
}
