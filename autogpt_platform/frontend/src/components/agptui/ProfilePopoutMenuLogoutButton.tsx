"use client";

import { logout } from "@/app/login/actions";
import { IconLogOut } from "@/components/ui/icons";

export const ProfilePopoutMenuLogoutButton = () => {
  return (
    <div
      className="inline-flex w-full items-center justify-start gap-2.5"
      onClick={() => logout()}
      role="button"
      tabIndex={0}
    >
      <div className="relative h-6 w-6">
        <IconLogOut className="h-6 w-6" />
      </div>
      <div className="font-sans text-base font-medium leading-normal text-neutral-800 dark:text-neutral-200">
        Log out
      </div>
    </div>
  );
};
