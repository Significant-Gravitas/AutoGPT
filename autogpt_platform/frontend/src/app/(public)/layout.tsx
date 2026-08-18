import { ReactNode } from "react";

interface Props {
  children: ReactNode;
}

export default function PublicLayout({ children }: Props) {
  return <main className="flex h-[100dvh] w-full flex-col">{children}</main>;
}
