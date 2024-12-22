import { ReactNode } from "react";

interface Props {
  children: ReactNode;
};

export default function AuthCard({
  children
}: Props) {
  return (
    <div className="flex w-[32rem] h-[80vh] items-center justify-center">
      <div className="w-full max-w-md bg-white rounded-lg p-6 shadow-md">
        {children}
      </div>
    </div>
  );
}
