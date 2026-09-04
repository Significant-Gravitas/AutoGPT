import { Avatar, AvatarFallback } from "@/components/atoms/Avatar/Avatar";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { Robot01Icon } from "@hugeicons/core-free-icons";
import Link from "next/link";
import { ROW_CLASS, ROW_LINK_CLASS } from "../helpers";

export function AutopilotRow() {
  return (
    <div className={`group ${ROW_CLASS} transition-colors hover:bg-zinc-50`}>
      <Link
        href="/copilot"
        aria-label="Open Autopilot"
        className={ROW_LINK_CLASS}
      />

      <Avatar className="relative h-9 w-9 shrink-0">
        <AvatarFallback>
          <Icon icon={Robot01Icon} size={18} />
        </AvatarFallback>
      </Avatar>

      <div className="relative min-w-0 flex-1">
        <Text variant="body-medium" className="truncate">
          Autopilot
        </Text>
        <Text variant="small" className="truncate text-zinc-500">
          Generalist — runs the shop
        </Text>
      </div>

      <div className="relative shrink-0 lg:opacity-0 lg:transition-opacity lg:group-focus-within:opacity-100 lg:group-hover:opacity-100">
        <Button
          as="NextLink"
          href="/copilot"
          variant="ghost"
          size="small"
          className="hidden text-zinc-600 lg:inline-flex"
        >
          Chat
        </Button>
      </div>
    </div>
  );
}
