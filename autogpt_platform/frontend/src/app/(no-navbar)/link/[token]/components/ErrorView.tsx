import { Text } from "@/components/atoms/Text/Text";
import { AuthCard } from "@/components/auth/AuthCard";
import { LinkBreak } from "@phosphor-icons/react";

interface Props {
  message: string;
}

export function ErrorView({ message }: Props) {
  return (
    <AuthCard title="Setup failed">
      <div className="flex w-full flex-col items-center gap-6">
        <div className="flex h-16 w-16 items-center justify-center rounded-full bg-red-100">
          <LinkBreak size={40} weight="bold" className="text-red-600" />
        </div>
        <Text
          variant="body-medium"
          className="text-center text-muted-foreground"
        >
          {message}
        </Text>
        <Text variant="small" className="text-center text-muted-foreground">
          Go back to your chat and ask the bot for a new setup link.
        </Text>
      </div>
    </AuthCard>
  );
}
