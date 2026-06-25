"use client";
import { SignOutIcon } from "@phosphor-icons/react";
import { useRouter } from "next/navigation";
import { AccountMenuRow } from "./AccountMenuRow";

export function AccountLogoutOption() {
  const router = useRouter();

  function handleLogout() {
    router.replace("/logout");
  }

  return (
    <AccountMenuRow
      as="button"
      destructive
      label="Log out"
      icon={
        <SignOutIcon className="h-[18px] w-[18px] shrink-0" weight="bold" />
      }
      onClick={handleLogout}
    />
  );
}
