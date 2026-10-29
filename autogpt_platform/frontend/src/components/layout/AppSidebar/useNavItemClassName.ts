import { cn } from "@/lib/utils";

export function useNavItemClassName() {
  return cn(
    "h-auto rounded-xl p-2 pl-3 font-normal data-[active=true]:font-normal group-data-[collapsible=icon]:!p-1.5 hover:!bg-surface-hover [&>svg]:size-4 group-data-[collapsible=icon]:[&>svg]:size-4.5 max-md:gap-3 max-md:py-2.5 max-md:text-base max-md:[&>svg]:size-5",
    "data-[active=true]:!bg-surface-active data-[active=true]:hover:!bg-surface-active",
  );
}
