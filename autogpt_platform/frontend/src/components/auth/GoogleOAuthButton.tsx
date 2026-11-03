import Image from "next/image";
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
      variant="secondary"
      className="w-full"
      onClick={onClick}
      disabled={disabled}
      loading={isLoading}
    >
      <Image src="/google-logo.svg" alt="Google" width={20} height={20} />
      {isLoading ? "Connecting..." : "Continue with Google"}
    </Button>
  );
}
