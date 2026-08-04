"use client";
import { SignOutIcon } from "@phosphor-icons/react";
import { useRouter } from "next/navigation";
import { AccountMenuRow } from "./AccountMenuRow";

interface Props {
  weight?: "bold" | "regular";
}

export function AccountLogoutOption({ weight = "bold" }: Props) {
  const router = useRouter();

  function handleLogout() {
    router.replace("/logout");
  }

  return (
    <AccountMenuRow
      as="button"
      destructive
      label="Log out"
      newLayout={weight === "regular"}
      icon={
        <SignOutIcon className="h-[18px] w-[18px] shrink-0" weight={weight} />
      }
      onClick={handleLogout}
    />
  );
}
