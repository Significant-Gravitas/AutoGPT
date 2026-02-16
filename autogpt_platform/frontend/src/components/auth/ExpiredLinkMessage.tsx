import { Button } from "../atoms/Button/Button";
import { Link } from "../atoms/Link/Link";
import { Text } from "../atoms/Text/Text";

interface Props {
  onSendNewLink: () => void;
  isLoading?: boolean;
  linkSent?: boolean;
}

export function ExpiredLinkMessage({
  onSendNewLink,
  isLoading = false,
  linkSent = false,
}: Props) {
  return (
    <div className="flex flex-col items-center gap-6">
      <Text variant="h3" className="text-center">
        Your reset password link has expired or has already been used
      </Text>
      <Text variant="body-medium" className="text-center text-muted-foreground">
        Click below to recover your password. A new link will be sent to your
        email.
      </Text>
      <Button
        variant="primary"
        onClick={onSendNewLink}
        loading={isLoading}
        disabled={linkSent}
        className="w-full"
      >
        {linkSent ? "Check Your Email" : "Send Me a New Link"}
      </Button>
      <div className="flex items-center gap-1">
        <Text variant="body-small" className="text-muted-foreground">
          Already have access?
        </Text>
        <Link href="/login" variant="secondary">
          Log in here
        </Link>
      </div>
    </div>
  );
}
