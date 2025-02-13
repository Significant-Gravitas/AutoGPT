import Link from "next/link";

type OnboardingButtonProps = {
  className?: string;
  children?: React.ReactNode;
  disabled?: boolean;
  onClick?: () => void;
  href?: string;
};

export default function OnboardingButton({
  className,
  children,
  disabled,
  onClick,
  href,
}: OnboardingButtonProps) {
  const buttonClasses = `
    ${className}
    font-geist text-white text-sm font-medium
    inline-flex justify-center items-center
    h-12 min-w-[100px] rounded-full py-3 px-5 gap-2.5
    transition-colors duration-200
    ${
      disabled
        ? "bg-zinc-300 cursor-not-allowed"
        : "bg-zinc-700 hover:bg-zinc-800"
    }
  `;

  if (href && !disabled) {
    return (
      <Link href={href} className={buttonClasses}>
        {children}
      </Link>
    );
  }

  return (
    <button onClick={onClick} disabled={disabled} className={buttonClasses}>
      {children}
    </button>
  );
}
