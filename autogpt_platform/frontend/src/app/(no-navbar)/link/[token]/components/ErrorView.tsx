import { Text } from "@/components/atoms/Text/Text";
import { AuthCard } from "@/components/auth/AuthCard";
import { Unlink01Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

interface Props {
  message: string;
}

export function ErrorView({ message }: Props) {
  return (
    <AuthCard title="Setup failed">
      <div className="flex w-full flex-col items-center gap-6">
        <div className="flex h-16 w-16 items-center justify-center rounded-full bg-error-surface text-error-foreground">
          <Icon icon={Unlink01Icon} size={40} />
        </div>
        <Text variant="body-medium" tone="muted" className="text-center">
          {message}
        </Text>
        <Text variant="small" tone="muted" className="text-center">
          Go back to your chat and ask the bot for a new setup link.
        </Text>
      </div>
    </AuthCard>
  );
}
