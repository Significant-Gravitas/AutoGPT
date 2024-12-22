import { ReactNode } from "react";

interface Props {
  children: ReactNode;
}

export default function AuthHeader({
  children
}: Props) {
  return (
    <div className="mb-8 text-slate-950 text-2xl font-semibold leading-normal">
      {children}
    </div>
  )
}
