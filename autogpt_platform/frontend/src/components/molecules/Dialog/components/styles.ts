import { cn } from "@/lib/utils";

// House sizing on top of Kobra's dialog and drawer panels (Tailwind class
// strings merged after Kobra's own, so these win).
const title = "font-poppins text-base md:text-lg leading-none";

export const modalStyles = {
  title,
  content:
    "flex max-h-[95vh] min-w-[40vw] max-w-[60vw] flex-col gap-0 p-6 sm:max-w-[60vw]",
};

// Compact variant: dense neutral dialog for in-app forms — smaller radius,
// tighter padding, sans title.
export const compactStyles = {
  content: "rounded-xl p-5",
  title: "font-sans text-base font-medium leading-6 text-popover-foreground",
  header: "pb-4",
  close: "right-3 top-3",
  // Bottom sheet keeps its top-only radius; only the padding tightens.
  drawerContent: "px-5 pb-5",
};

export const drawerStyles = {
  title: cn(title, "font-semibold text-popover-foreground"),
  content: "mt-0 h-auto max-h-[90vh] min-h-0 rounded-t-3xl px-6 pb-6",
};
