import { ReactNode } from "react";

interface Props {
  children: ReactNode;
}

export default function AuthCard({ children }: Props) {
  return (
    <div className="flex h-[80vh] w-[32rem] items-center justify-center">
      <div className="w-full max-w-md rounded-lg bg-white p-6 shadow-md">
        {children}
      </div>
    </div>
  );
}
