import Image from "next/image";
import { Button } from "../atoms/Button/Button";

interface Props {
  onClick: () => void;
  isLoading?: boolean;
  disabled?: boolean;
}

export function MicrosoftOAuthButton({ onClick, isLoading, disabled }: Props) {
  return (
    <Button
      type="button"
      variant="secondary"
      className="w-full gap-3"
      onClick={onClick}
      disabled={disabled}
      loading={isLoading}
    >
      <Image
        src="/integrations/microsoft.webp"
        alt="Microsoft"
        width={20}
        height={20}
      />
      {isLoading ? "Connecting..." : "Continue with Microsoft"}
    </Button>
  );
}
