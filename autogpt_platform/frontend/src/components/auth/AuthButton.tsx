import { ReactNode } from "react";
import { FaSpinner } from "react-icons/fa";
import { Button } from "../ui/button";

interface Props {
  children?: ReactNode;
  isLoading?: boolean;
  disabled?: boolean;
  type?: "button" | "submit" | "reset";
  onClick?: () => void;
}

export default function AuthButton({
  children,
  isLoading = false,
  disabled = false,
  type = "button",
  onClick,
}: Props) {
  return (
    <Button
      className="mt-2 w-full px-4 py-2 text-zinc-800"
      variant="outline"
      type={type}
      disabled={isLoading || disabled}
      onClick={onClick}
    >
      {isLoading ? (
        <FaSpinner className="animate-spin" />
      ) : (
        <div className="text-sm font-medium">{children}</div>
      )}
    </Button>
  );
}
