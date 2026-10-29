import { ReactNode } from "react";

import { AdminShell } from "./components/AdminShell";

export default function AdminLayout({ children }: { children: ReactNode }) {
  return <AdminShell>{children}</AdminShell>;
}
