import { ReactNode } from "react";
import { Button } from "../ui/button";
import { FaSpinner } from "react-icons/fa";

interface Props {
  children?: ReactNode;
  onClick: () => void;
  isLoading?: boolean;
  type?: "button" | "submit" | "reset";
}

export default function AuthButton({
  children,
  onClick,
  isLoading = false,
  type = "button"
}: Props) {
  return (
    <Button
      className="mt-2 self-stretch w-full px-4 py-2 bg-slate-900 rounded-md"
      type={type}
      disabled={isLoading}
      onClick={onClick}
    >
      {isLoading ? <FaSpinner className="animate-spin" /> :
        <div className="text-slate-50 text-sm font-medium leading-normal">{children}</div>}
    </Button>
  );
}
