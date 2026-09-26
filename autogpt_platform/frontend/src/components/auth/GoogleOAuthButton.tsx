import Image from "next/image";
import { Button } from "../atoms/Button/Button";
import { Link } from "../atoms/Link/Link";
import { Text } from "../atoms/Text/Text";

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
    <div className="flex w-full flex-col gap-2">
      <Button
        type="button"
        variant="secondary"
        className="w-full gap-3"
        onClick={onClick}
        disabled={disabled}
        loading={isLoading}
      >
        <Image src="/google-logo.svg" alt="Google" width={20} height={20} />
        {isLoading ? "Connecting..." : "Continue with Google"}
      </Button>
      <Text variant="small" className="px-4 text-center !text-slate-500">
        By continuing with Google, you agree to the{" "}
        <Link
          href="https://agpt.co/legal/platform-terms-of-use"
          isExternal
          variant="secondary"
          className="text-xs font-normal leading-[1.125rem] !text-slate-500"
        >
          Terms of Use
        </Link>{" "}
        and acknowledge the{" "}
        <Link
          href="https://agpt.co/legal/platform-privacy-policy"
          isExternal
          variant="secondary"
          className="text-xs font-normal leading-[1.125rem] !text-slate-500"
        >
          Privacy Policy
        </Link>
        .
      </Text>
    </div>
  );
}
