import { Text } from "@/components/atoms/Text/Text";
import { AuthCard } from "@/components/auth/AuthCard";
import { Loading03Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

interface Props {
  title?: string;
  message?: string;
}

export function LoadingView({
  title = "Setting up AutoGPT",
  message = "Loading...",
}: Props) {
  return (
    <AuthCard title={title}>
      <div className="flex flex-col items-center gap-4">
        <div className="animate-spin text-foreground-strong">
          <Icon icon={Loading03Icon} size={48} />
        </div>
        <Text variant="body-medium" tone="muted">
          {message}
        </Text>
      </div>
    </AuthCard>
  );
}
