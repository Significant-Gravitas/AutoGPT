import { Text } from "@/components/atoms/Text/Text";
import { AuthCard } from "@/components/auth/AuthCard";
import { Spinner } from "@phosphor-icons/react";

interface Props {
  title?: string;
  message?: string;
}

export function LoadingView({
  title = "Setting up AutoPilot",
  message = "Loading...",
}: Props) {
  return (
    <AuthCard title={title}>
      <div className="flex flex-col items-center gap-4">
        <Spinner size={48} className="animate-spin text-primary" />
        <Text variant="body-medium" className="text-muted-foreground">
          {message}
        </Text>
      </div>
    </AuthCard>
  );
}
