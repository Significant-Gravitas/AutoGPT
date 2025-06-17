import { FaGoogle, FaSpinner } from "react-icons/fa";
import { Button } from "../ui/button";

interface GoogleOAuthButtonProps {
  onClick: () => void;
  isLoading?: boolean;
  disabled?: boolean;
}

export default function GoogleOAuthButton({
  onClick,
  isLoading = false,
  disabled = false,
}: GoogleOAuthButtonProps) {
  return (
    <Button
      type="button"
      className="w-full border bg-zinc-700 py-2 text-white disabled:opacity-50"
      disabled={isLoading || disabled}
      onClick={onClick}
    >
      {isLoading ? (
        <FaSpinner className="mr-2 h-4 w-4 animate-spin" />
      ) : (
        <FaGoogle className="mr-2 h-4 w-4" />
      )}
      <span className="text-sm font-medium">
        {isLoading ? "Signing in..." : "Continue with Google"}
      </span>
    </Button>
  );
}
