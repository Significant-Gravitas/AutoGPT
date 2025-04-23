import { ReactNode } from "react";

interface Props {
  children: ReactNode;
}

export default function AuthHeader({ children }: Props) {
  return (
    <div className="mb-8 text-2xl font-semibold leading-normal text-slate-950">
      {children}
    </div>
  );
}
