import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { PlugSocketIcon } from "@hugeicons/core-free-icons";

interface Props {
  names: string[];
  onConnect: () => void;
}

/** Offered after the install has already succeeded, so it reads as the next
 *  thing worth doing rather than something that went wrong. */
export function ConnectStep({ names, onConnect }: Props) {
  const providers = new Intl.ListFormat("en", {
    style: "long",
    type: "conjunction",
  }).format(names);

  return (
    <div
      className="mt-6 flex flex-wrap items-center gap-3 rounded-xl border border-zinc-200 bg-white p-4"
      data-testid="skill-connect-step"
    >
      <Icon
        icon={PlugSocketIcon}
        size={18}
        className="text-zinc-500"
        aria-hidden
      />
      <Text variant="small" className="!text-zinc-600">
        Its steps use {providers}. Connect when you first need it.
      </Text>
      <Button
        variant="secondary"
        size="small"
        onClick={onConnect}
        className="ml-auto"
      >
        Connect
      </Button>
    </div>
  );
}
