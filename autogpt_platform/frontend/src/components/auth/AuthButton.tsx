import { ReactNode } from "react";
import { Button } from "../ui/button";
import { FaSpinner } from "react-icons/fa";

interface Props {
  children?: ReactNode;
  onClick: () => void;
  isLoading?: boolean;
  disabled?: boolean;
  type?: "button" | "submit" | "reset";
}

export default function AuthButton({
  children,
  onClick,
  isLoading = false,
  disabled = false,
  type = "button",
}: Props) {
  return (
    <Button
      className="mt-2 w-full self-stretch rounded-md bg-slate-900 px-4 py-2"
      type={type}
      disabled={isLoading || disabled}
      onClick={onClick}
    >
      {isLoading ? (
        <FaSpinner className="animate-spin" />
      ) : (
        <div className="text-sm font-medium leading-normal text-slate-50">
          {children}
        </div>
      )}
    </Button>
  );
}
