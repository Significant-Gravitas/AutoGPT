import { cn } from "@/lib/utils";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";

// The stronger active state ships with the brain-dump experience.
export function useNavItemClassName() {
  const isBrainDumpEnabled = useGetFlag(Flag.ONBOARDING_BRAIN_DUMP);
  return cn(
    "h-auto rounded-xl p-2 pl-3 font-normal data-[active=true]:font-normal group-data-[collapsible=icon]:!p-1.5 hover:!bg-zinc-100 [&>svg]:size-4 group-data-[collapsible=icon]:[&>svg]:size-4.5",
    isBrainDumpEnabled
      ? "data-[active=true]:!bg-zinc-200 data-[active=true]:hover:!bg-zinc-200"
      : "data-[active=true]:!bg-zinc-100",
  );
}
