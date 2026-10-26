import { Icon } from "@/components/atoms/Icon/Icon";
import { Unlink04Icon } from "@hugeicons/core-free-icons";
import { formatUnsupported } from "./helpers";

interface Props {
  services: string[] | undefined;
}

/** Under a suggested task that relies on a tool Otto can't connect yet, so
 *  picking it is an informed choice rather than a dead end found later. */
export function UnsupportedNote({ services }: Props) {
  const label = formatUnsupported(services);
  if (!label) return null;
  return (
    <span className="mt-0.5 flex items-start gap-1 text-sm text-foreground-tertiary">
      <Icon icon={Unlink04Icon} size={14} className="mt-[3px] shrink-0" />
      <span className="text-pretty">{label}</span>
    </span>
  );
}
