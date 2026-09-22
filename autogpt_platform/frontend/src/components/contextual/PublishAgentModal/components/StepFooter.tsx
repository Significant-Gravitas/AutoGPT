import { ReactNode } from "react";

type Props = {
  primary: ReactNode;
  secondary?: ReactNode;
};

export function StepFooter({ primary, secondary }: Props) {
  return (
    <div className="mt-2 flex flex-col-reverse gap-3 pt-5 sm:flex-row sm:items-center sm:justify-end">
      {secondary}
      {primary}
    </div>
  );
}
