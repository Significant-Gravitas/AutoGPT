"use client";
import { useRouter } from "next/navigation";
import { AccountMenuRow } from "./AccountMenuRow";
import { Logout03Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

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
      icon={<Icon icon={Logout03Icon} className="h-[18px] w-[18px] shrink-0" />}
      onClick={handleLogout}
    />
  );
}
