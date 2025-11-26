import { Navbar } from "@/components/layout/Navbar/Navbar";
import { AdminImpersonationBanner } from "./admin/components/AdminImpersonationBanner";
import { ReactNode } from "react";

export default function PlatformLayout({ children }: { children: ReactNode }) {
  return (
    <main className="flex h-screen w-full flex-col">
      <Navbar />
      <AdminImpersonationBanner />
      <section className="flex-1">{children}</section>
    </main>
  );
}
