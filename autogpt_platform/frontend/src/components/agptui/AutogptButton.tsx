import { PlayIcon, Loader2Icon } from "lucide-react";
import { Button } from "../ui/button";
import { cva, type VariantProps } from "class-variance-authority";
import { cn } from "@/lib/utils";

const buttonVariants = cva("inline-flex items-center justify-center", {
  variants: {
    variant: {
      default:
        "bg-zinc-700 text-white hover:bg-zinc-800 disabled:bg-zinc-300 border-none",
      secondary:
        "bg-zinc-200 text-zinc-800 hover:bg-zinc-300 disabled:bg-zinc-50 disabled:text-zinc-300 border-none",
      destructive:
        "bg-red-500 text-white hover:bg-red-600 disabled:bg-zinc-300 disabled:text-white border-none",
      outline:
        "bg-white text-zinc-800 hover:bg-zinc-100 border border-zinc-700 disabled:bg-white disabled:border-zinc-300 disabled:text-zinc-300",
      ghost:
        "bg-transparent text-zinc-800 hover:bg-zinc-100  disabled:text-zinc-300 disabled:bg-transparent border-none",
      link: "bg-transparent text-zinc-800 hover:underline border-none hover:bg-transparent",
    },
  },
  defaultVariants: {
    variant: "default",
  },
});

interface AutogptButtonProps extends VariantProps<typeof buttonVariants> {
  icon?: boolean;
  children: React.ReactNode;
  isDisabled?: boolean;
  isLoading?: boolean;
  [key: string]: any; // Allow any additional props
}

const AutogptButton = ({
  icon = false,
  children,
  variant,
  isDisabled = false,
  isLoading = false,
  ...props
}: AutogptButtonProps) => {
  return (
    <Button
      className={cn(
        "h-12 space-x-1.5 rounded-[3rem] px-4 py-3 shadow-none",
        buttonVariants({ variant }),
        isDisabled && "bg-red-500",
        isLoading && "bg-[#3F3F4680] text-white",
      )}
      disabled={isDisabled || isLoading}
      variant={variant}
      {...props}
    >
      {isLoading ? (
        <Loader2Icon className="h-4 w-4 animate-spin" />
      ) : (
        icon && <PlayIcon className="h-5 w-5" />
      )}
      <p className="font-sans text-sm font-medium">{children}</p>
    </Button>
  );
};

export default AutogptButton;
