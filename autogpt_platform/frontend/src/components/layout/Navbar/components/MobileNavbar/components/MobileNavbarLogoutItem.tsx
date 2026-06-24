"use client";

import { IconType } from "@/components/__legacy__/ui/icons";
import { useRouter } from "next/navigation";
import { getAccountMenuOptionIcon } from "../../../helpers";

interface Props {
  icon: IconType;
  text: string;
}

export function MobileNavbarLogoutItem({ icon, text }: Props) {
  const router = useRouter();

  async function handleLogout() {
    router.replace("/logout");
  }

  return (
    <button className="w-full" onClick={handleLogout} type="button">
      <div className="inline-flex w-full items-center justify-start gap-4 py-2 hover:rounded hover:bg-[#e0e0e0]">
        {getAccountMenuOptionIcon(icon)}
        <div className="relative">
          <div className="font-sans text-base font-normal leading-7 text-[#474747]">
            {text}
          </div>
        </div>
      </div>
    </button>
  );
}
