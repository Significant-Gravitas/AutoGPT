import { ReactNode } from "react";

interface Props {
  title: string;
  children: ReactNode;
}

export function ProfileSection({ title, children }: Props) {
  return (
    <div className="relative mt-8">
      <div className="mb-2.5 text-xs font-medium uppercase tracking-[0.14em] text-zinc-400">
        {title}
      </div>
      {children}
    </div>
  );
}
