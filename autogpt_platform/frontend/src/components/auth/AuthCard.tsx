import { cn } from "@/lib/utils";
import { ReactNode } from "react";
import { Card } from "../atoms/Card/Card";
import { Link } from "../atoms/Link/Link";
import { Text } from "../atoms/Text/Text";

interface BottomTextProps {
  text: string;
  link?: { text: string; href: string };
  className?: string;
}

AuthCard.BottomText = function BottomText({
  text,
  link,
  className,
}: BottomTextProps) {
  return (
    <div
      className={cn(
        className,
        "mt-8 inline-flex w-full items-center justify-center",
      )}
    >
      <span className="text-sm font-medium leading-normal text-slate-950">
        {text}
      </span>
      {link ? (
        <Link
          href={link.href}
          className="ml-1 text-sm font-medium leading-normal text-slate-950 underline"
        >
          {link.text}
        </Link>
      ) : null}
    </div>
  );
};

interface Props {
  children: ReactNode;
  title: string;
  className?: string;
}

export function AuthCard({ children, title }: Props) {
  return (
    <Card className="mx-auto flex min-h-[50vh] w-full max-w-[32rem] flex-col items-center justify-center gap-8">
      <Text variant="h3" as="h2">
        {title}
      </Text>
      {children}
    </Card>
  );
}
