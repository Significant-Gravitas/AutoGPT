import { Button } from "../atoms/Button/Button";
import { Link } from "../atoms/Link/Link";
import { Text } from "../atoms/Text/Text";

interface Props {
  onRequestNewLink: () => void;
}

export function ExpiredLinkMessage({ onRequestNewLink }: Props) {
  return (
    <div className="flex flex-col items-center gap-6">
      <Text variant="h3" className="text-center">
        Your reset password link has expired or has already been used
      </Text>
      <Text variant="body-medium" className="text-center text-muted-foreground">
        Click below to request a new password reset link.
      </Text>
      <Button variant="primary" onClick={onRequestNewLink} className="w-full">
        Request a New Link
      </Button>
      <div className="flex items-center gap-1">
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
