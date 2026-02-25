"use client";
import { IconLogOut } from "@/components/__legacy__/ui/icons";
import { useRouter } from "next/navigation";

export function AccountLogoutOption() {
  const router = useRouter();

  async function handleLogout() {
    router.replace("/logout");
  }

  return (
    <button
      className="inline-flex w-full items-center justify-start gap-2.5"
      onClick={handleLogout}
      type="button"
    >
      <div className="relative h-4 w-4">
        <IconLogOut />
      </div>
      <div className="font-sans text-base font-medium leading-normal text-neutral-800 dark:text-neutral-200">
        Log out
      </div>
    </button>
  );
}
