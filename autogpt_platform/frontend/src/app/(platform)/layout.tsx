import { Navbar } from "@/components/layout/Navbar/Navbar";
import { NetworkStatusMonitor } from "@/services/network-status/NetworkStatusMonitor";
import { ReactNode } from "react";
import { AdminImpersonationBanner } from "./admin/components/AdminImpersonationBanner";

export default function PlatformLayout({ children }: { children: ReactNode }) {
  return (
    <main className="flex h-screen w-full flex-col">
      <NetworkStatusMonitor />
      <Navbar />
      <AdminImpersonationBanner />
      <section className="flex-1">{children}</section>
    </main>
  );
}
