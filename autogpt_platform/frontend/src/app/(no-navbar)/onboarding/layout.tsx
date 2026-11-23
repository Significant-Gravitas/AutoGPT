import { ServerUserSeed } from "@/components/auth/ServerUserSeed";
import { getServerUser } from "@/lib/auth/server/getServerUser";
import { ReactNode } from "react";

export default async function OnboardingLayout({
  children,
}: {
  children: ReactNode;
}) {
  const { user } = await getServerUser();
  return (
    <div className="relative flex min-h-screen w-full flex-col bg-surface-hover">
      <ServerUserSeed user={user} />
      <main className="flex w-full flex-1 flex-col items-center justify-center">
        {children}
      </main>
    </div>
  );
}
