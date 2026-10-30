import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useState } from "react";
import { isWelcomeDone } from "./components/ExpertWelcome/helpers";

/** The welcome chat opens with the page that the create dialog navigates to
 *  (`?welcome=1`), once per expert. */
export function useExpertWelcomeState(expertId: string) {
  const searchParams = useSearchParams();
  const router = useRouter();
  const pathname = usePathname();
  const [isOpen, setIsOpen] = useState(
    () => searchParams.get("welcome") === "1" && !isWelcomeDone(expertId),
  );

  function close() {
    setIsOpen(false);
    if (searchParams.get("welcome"))
      router.replace(pathname, { scroll: false });
  }

  return { isWelcomeOpen: isOpen, closeWelcome: close };
}
