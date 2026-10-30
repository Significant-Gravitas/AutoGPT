"use client";

import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { jobTitleForRoleLink } from "./helpers";
import { RaiseExpertDialog } from "./RaiseExpertDialog";

/** Opens the create dialog for a `?raise=1` link, which is where `/raise`
 *  and Otto's "create one" suggestions now land. */
export function RaiseFromLink() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const pathname = usePathname();
  const isOpen = searchParams.get("raise") === "1";

  function close() {
    router.replace(pathname, { scroll: false });
  }

  return (
    <RaiseExpertDialog
      open={isOpen}
      onClose={close}
      initialJobTitle={jobTitleForRoleLink(searchParams.get("role"))}
    />
  );
}
