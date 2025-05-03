import { LoaderCircle } from "lucide-react";

export default function Spinner({ className }: { className?: string }) {
  const spinnerClasses = `mr-2 h-16 w-16 animate-spin ${className || ""}`;

  return (
    <div className="flex h-[80vh] items-center justify-center">
      <LoaderCircle className={spinnerClasses} />
    </div>
  );
}
