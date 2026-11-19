import { Alert, AlertDescription } from "@/components/molecules/Alert/Alert";
import { Icon } from "@/components/atoms/Icon/Icon";
import { UserBlock01Icon } from "@hugeicons/core-free-icons";
import { type AdminUser, formatDateTime, formatTimeUntil } from "../helpers";

interface Props {
  user: AdminUser;
}

export function BannedAlert({ user }: Props) {
  const until = user.banExpires
    ? `until ${formatDateTime(user.banExpires)} (${formatTimeUntil(user.banExpires)})`
    : "permanently";

  return (
    <Alert variant="error" icon={BannedIcon}>
      <AlertDescription className="break-words">
        <strong className="font-medium">Banned</strong> · {until}
        {user.banReason ? <> · &ldquo;{user.banReason}&rdquo;</> : null}
      </AlertDescription>
    </Alert>
  );
}

function BannedIcon({ className }: { className?: string }) {
  return <Icon icon={UserBlock01Icon} className={className} aria-hidden />;
}
