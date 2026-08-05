import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Settings01Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

export function AgentSettingsButton() {
  return (
    <Button
      variant="ghost"
      size="small"
      className="m-0 min-w-0 rounded-full p-0 px-1"
      aria-label="Agent Settings"
    >
      <Icon icon={Settings01Icon} size={18} className="text-zinc-600" />
      <Text variant="small">Agent Settings</Text>
    </Button>
  );
}
