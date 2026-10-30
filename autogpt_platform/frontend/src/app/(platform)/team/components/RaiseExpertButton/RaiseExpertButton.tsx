"use client";

import { Button } from "@/components/atoms/Button/Button";
import { useState, type ReactNode } from "react";
import { RaiseExpertDialog } from "../RaiseExpertDialog/RaiseExpertDialog";

interface Props {
  variant?: "primary" | "secondary";
  children?: ReactNode;
}

export function RaiseExpertButton({
  variant = "secondary",
  children = "Create an Expert",
}: Props) {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <>
      <Button
        type="button"
        variant={variant}
        size="small"
        onClick={() => setIsOpen(true)}
      >
        {children}
      </Button>
      <RaiseExpertDialog open={isOpen} onClose={() => setIsOpen(false)} />
    </>
  );
}
