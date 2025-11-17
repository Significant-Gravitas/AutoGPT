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
        This password reset link has expired or already been used
      </Text>
      <div className="flex flex-col gap-4 text-center">
        <Text variant="large" className="text-center text-muted-foreground">
          Don&apos;t worry – this can happen if the link is opened more than
          once or has timed out.
        </Text>
        <Text variant="large" className="text-center text-muted-foreground">
          Click below to request a new password reset link.
        </Text>
      </div>
      <Button
        variant="primary"
        onClick={onSendNewLink}
        loading={isLoading}
        disabled={linkSent}
        className="w-full max-w-sm"
      >
        {linkSent ? "Link Sent!" : "Send Me a New Link"}
      </Button>
      <div className="mt-2 flex items-center gap-1">
        <Text variant="small" className="text-muted-foreground">
          Already have access?
        </Text>
        <Link href="/login" variant="secondary">
          Log in here
        </Link>
      </div>
    </div>
  );
}
