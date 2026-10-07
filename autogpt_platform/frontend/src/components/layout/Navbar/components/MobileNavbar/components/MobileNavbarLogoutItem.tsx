"use client";

import { useRouter } from "next/navigation";
import { getAccountMenuOptionIcon, IconType } from "../../../helpers";

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
      <div className="inline-flex w-full items-center justify-start gap-4 py-2 hover:rounded-sm hover:bg-zinc-200">
        {getAccountMenuOptionIcon(icon)}
        <div className="relative">
          <div className="font-sans text-base leading-7 font-normal text-zinc-800">
            {text}
          </div>
        </div>
      </div>
    </button>
  );
}
