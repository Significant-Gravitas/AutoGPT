import { ChatDrawer } from "@/components/contextual/Chat/ChatDrawer";
import { Navbar } from "@/components/layout/Navbar/Navbar";
import { ReactNode } from "react";
import { AdminImpersonationBanner } from "./admin/components/AdminImpersonationBanner";

export default function PlatformLayout({ children }: { children: ReactNode }) {
  return (
    <main className="flex h-screen w-full flex-row overflow-hidden">
      <div className="flex min-w-0 flex-1 flex-col">
        <Navbar />
        <AdminImpersonationBanner />
        <section className="flex-1 overflow-auto">{children}</section>
      </div>
      <ChatDrawer />
    </main>
  );
}
