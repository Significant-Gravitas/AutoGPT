import { Avatar, AvatarFallback } from "@/components/atoms/Avatar/Avatar";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Robot01Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

export function AutopilotCard() {
  return (
    <div className="flex flex-col gap-3 rounded-[1.75rem] border border-zinc-200 bg-white p-5">
      <div className="flex items-center gap-3">
        <Avatar className="h-12 w-12">
          <AvatarFallback>
            <Icon icon={Robot01Icon} size={24} />
          </AvatarFallback>
        </Avatar>
        <div className="min-w-0 flex-1">
          <Text variant="large-medium">Autopilot</Text>
          <Text variant="small" className="text-zinc-500">
            Head of AI — runs the team
          </Text>
        </div>
      </div>
      <Text variant="body">
        Owns the outcome, assigns work, answers expert questions, and asks you
        only for decisions it cannot safely make.
      </Text>
      <div className="mt-auto flex gap-2">
        <Button as="NextLink" href="/copilot" variant="secondary" size="small">
          Chat
        </Button>
      </div>
    </div>
  );
}
