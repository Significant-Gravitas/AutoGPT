import { ChatDrawer } from "@/components/contextual/Chat/ChatDrawer";
import { Navbar } from "@/components/layout/Navbar/Navbar";
import { ReactNode } from "react";
import { AdminImpersonationBanner } from "./admin/components/AdminImpersonationBanner";

export default function PlatformLayout({ children }: { children: ReactNode }) {
  return (
    <main className="flex h-screen w-full flex-col overflow-hidden">
      <Navbar />
      <AdminImpersonationBanner />
      <section className="flex min-h-0 flex-1 overflow-auto">
        {children}
      </section>
      <ChatDrawer />
    </main>
  );
}
