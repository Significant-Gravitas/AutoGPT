import { IconAutoGPTLogo } from "@/components/ui/icons";
import { Skeleton } from "@/components/ui/skeleton";

export function NavbarLoading() {
  return (
    <nav className="sticky top-0 z-40 hidden h-16 items-center rounded-bl-2xl rounded-br-2xl border border-white/50 bg-white/5 p-3 backdrop-blur-[26px] md:inline-flex">
      <div className="flex flex-1 items-center gap-6">
        <Skeleton className="h-4 w-20 bg-white/20" />
        <Skeleton className="h-4 w-16 bg-white/20" />
        <Skeleton className="h-4 w-12 bg-white/20" />
      </div>
      <div className="absolute left-1/2 top-1/2 h-10 w-[88.87px] -translate-x-1/2 -translate-y-1/2">
        <IconAutoGPTLogo className="h-full w-full" />
      </div>
      <div className="flex flex-1 items-center justify-end gap-4">
        <Skeleton className="h-8 w-8 rounded-full bg-white/20" />
        <Skeleton className="h-8 w-8 rounded-full bg-white/20" />
      </div>
    </nav>
  );
}
