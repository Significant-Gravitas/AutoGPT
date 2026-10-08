import { Kbd as KobraKbd } from "@/components/ui/kbd";
import { cn } from "@/lib/utils";

type KbdSize = "sm" | "md";

interface Props extends React.ComponentProps<"kbd"> {
  size?: KbdSize;
}

// House sm/md are 20/24px, which are Kobra's md/lg.
const kbdSizes: Record<KbdSize, React.ComponentProps<typeof KobraKbd>["size"]> =
  {
    sm: "md",
    md: "lg",
  };

export function Kbd({ className, size = "sm", ...props }: Props) {
  return (
    <KobraKbd
      size={kbdSizes[size]}
      className={cn("sentry-unmask", className)}
      {...props}
    />
  );
}
