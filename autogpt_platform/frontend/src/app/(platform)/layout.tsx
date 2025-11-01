import { Navbar } from "@/components/layout/Navbar/Navbar";
import { AdminImpersonationBanner } from "@/components/admin/AdminImpersonationBanner";
import { ReactNode } from "react";

export default function PlatformLayout({ children }: { children: ReactNode }) {
  return (
    <main className="flex h-screen w-full flex-col">
      <Navbar />
      <div className="px-4 pt-2">
        <AdminImpersonationBanner />
      </div>
      <section className="flex-1">{children}</section>
    </main>
  );
}
