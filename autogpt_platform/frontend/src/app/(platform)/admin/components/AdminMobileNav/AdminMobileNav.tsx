"use client";

import { SectionMobileNav } from "@/components/layout/SectionNav/SectionMobileNav";
import { useAdminSidebar } from "../AdminSidebar/useAdminSidebar";

export function AdminMobileNav() {
  const { items } = useAdminSidebar();
  return <SectionMobileNav items={items} sectionName="Admin" showBackLink />;
}
