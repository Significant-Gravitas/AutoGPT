import { GoogleLogo } from "@phosphor-icons/react/ssr";
import { Button } from "../atoms/Button/Button";

interface GoogleOAuthButtonProps {
  onClick: () => void;
  isLoading?: boolean;
  disabled?: boolean;
}

export function GoogleOAuthButton({
  onClick,
  isLoading,
  disabled,
}: GoogleOAuthButtonProps) {
  return (
    <Button
      type="button"
      variant="outline"
      className="w-full gap-3"
      onClick={onClick}
      disabled={disabled}
      loading={isLoading}
    >
      <GoogleLogo size={20} />
      {isLoading ? "Connecting..." : "Continue with Google"}
    </Button>
  );
}
