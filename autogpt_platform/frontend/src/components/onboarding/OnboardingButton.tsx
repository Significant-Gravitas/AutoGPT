import Link from "next/link";

type OnboardingButtonProps = {
  children?: React.ReactNode;
  disabled?: boolean;
  onClick?: () => void;
  href?: string;
 } & React.ButtonHTMLAttributes<HTMLButtonElement>;
 
 export default function OnboardingButton({ children, disabled, onClick, href, ...props }: OnboardingButtonProps) {
  const buttonClasses = `
    font-geist text-white
    h-12 min-w-[100px] rounded-full py-3 px-5 gap-2.5
    transition-colors duration-200
    ${disabled ?
      'bg-zinc-300 cursor-not-allowed' :
      'bg-zinc-700 hover:bg-zinc-800'
    }
  `;
 
  if (href) {
    return (
      <Link href={href} className={buttonClasses}>
        {children}
      </Link>
    );
  }
 
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={buttonClasses}
      {...props}
    >
      {children}
    </button>
  );
 }