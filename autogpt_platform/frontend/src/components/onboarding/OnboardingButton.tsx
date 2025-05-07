import { cn } from "@/lib/utils";
import Link from "next/link";
import { useCallback, useMemo, useState } from "react";
import Spinner from "../Spinner";

const variants = {
  default: "bg-zinc-700 hover:bg-zinc-800",
  violet: "bg-violet-600 hover:bg-violet-700",
};

type OnboardingButtonProps = {
  className?: string;
  variant?: keyof typeof variants;
  children?: React.ReactNode;
  loading?: boolean;
  disabled?: boolean;
  onClick?: () => void;
  href?: string;
  icon?: React.ReactNode;
};

export default function OnboardingButton({
  className,
  variant = "default",
  children,
  loading,
  disabled,
  onClick,
  href,
  icon,
}: OnboardingButtonProps) {
  const [internalLoading, setInternalLoading] = useState(false);
  const isLoading = loading !== undefined ? loading : internalLoading;

  const buttonClasses = useMemo(
    () =>
      cn(
        "font-sans text-white text-sm font-medium",
        "inline-flex justify-center items-center",
        "h-12 min-w-[100px] rounded-full py-3 px-5",
        "transition-colors duration-200",
        className,
        disabled ? "bg-zinc-300 cursor-not-allowed" : variants[variant],
      ),
    [disabled, variant, className],
  );

  const onClickInternal = useCallback(() => {
    setInternalLoading(true);
    if (onClick) {
      onClick();
    }
  }, [setInternalLoading, onClick]);

  if (href && !disabled) {
    return (
      <Link href={href} onClick={onClickInternal} className={buttonClasses}>
        {isLoading && <Spinner className="h-5 w-5" />}
        {icon && !isLoading && <>{icon}</>}
        {children}
      </Link>
    );
  }

  return (
    <button
      onClick={onClickInternal}
      disabled={disabled}
      className={buttonClasses}
    >
      {isLoading && <Spinner className="h-5 w-5" />}
      {icon && !isLoading && <>{icon}</>}
      {children}
    </button>
  );
}
