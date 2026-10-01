"use client";

import { usePlatformChrome } from "@/app/(platform)/PlatformChrome/usePlatformChrome";
import { ReactNode } from "react";

import { AdminClassicShell } from "./components/AdminClassicShell";
import { AdminNewShell } from "./components/AdminNewShell";

// Switcher between the classic admin shell (legacy sidebar under the top
// Navbar) and the new-layout shell (settings-style sidebar, no Navbar). The
// classic shell can be deleted wholesale once the new layout ships.
export default function AdminLayout({ children }: { children: ReactNode }) {
  const { isNewLayoutActive, isLayoutPending } = usePlatformChrome();

  // Neither shell is known yet (see usePlatformChrome): hold both back rather
  // than paint the legacy sidebar for a new-layout user and swap it out.
  if (isLayoutPending) {
    return null;
  }

  if (isNewLayoutActive) {
    return <AdminNewShell>{children}</AdminNewShell>;
  }

  return <AdminClassicShell>{children}</AdminClassicShell>;
}
