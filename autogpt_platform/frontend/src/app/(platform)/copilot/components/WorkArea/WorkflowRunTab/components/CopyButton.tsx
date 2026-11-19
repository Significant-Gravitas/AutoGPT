"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Copy01Icon, Tick02Icon } from "@hugeicons/core-free-icons";
import { useEffect, useState } from "react";

const COPIED_MS = 1500;

interface Props {
  text: string;
  label: string;
}

/** Copies a value; the icon turns to a check for a moment. */
export function CopyButton({ text, label }: Props) {
  const [isCopied, setIsCopied] = useState(false);

  useEffect(() => {
    if (!isCopied) return;
    const timer = setTimeout(() => setIsCopied(false), COPIED_MS);
    return () => clearTimeout(timer);
  }, [isCopied]);

  function copy() {
    navigator.clipboard
      .writeText(text)
      .then(() => setIsCopied(true))
      .catch(() => setIsCopied(false));
  }

  return (
    <Button
      variant="ghost-muted"
      size="xs"
      leadingIcon={isCopied ? Tick02Icon : Copy01Icon}
      aria-label={label}
      onClick={copy}
    >
      {isCopied ? "Copied" : "Copy"}
    </Button>
  );
}
