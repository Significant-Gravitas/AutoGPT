import { cn } from "@/lib/utils";

export function useNavItemClassName() {
  return cn(
    "h-auto gap-2 rounded-lg px-2 py-1.5 font-normal data-[active=true]:font-normal group-data-[collapsible=icon]:!p-2 hover:!bg-shell-hover [&>svg]:size-4 max-md:gap-3 max-md:py-2.5 max-md:text-body-lg max-md:[&>svg]:size-5",
    "data-[active=true]:!bg-shell-hover data-[active=true]:hover:!bg-shell-hover",
  );
}
