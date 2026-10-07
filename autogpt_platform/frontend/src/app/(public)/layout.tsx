import { ReactNode } from "react";

interface Props {
  children: ReactNode;
}

export default function PublicLayout({ children }: Props) {
  return <main className="flex h-dvh w-full flex-col">{children}</main>;
}
