import { motion } from "framer-motion";
import type { ReactNode, Ref } from "react";
import { cn } from "@/lib/utils";
import { PausedDraftNote } from "./PausedDraftNote";

interface Props {
  children: ReactNode;
  notice?: ReactNode;
  noticeRef?: Ref<HTMLDivElement>;
  paused?: boolean;
  className?: string;
}

/** Shared composer spacing keeps the floating notice outside the input row. */
export function ChatComposerFrame({
  children,
  notice,
  noticeRef,
  paused = false,
  className,
}: Props) {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.3 }}
      className={cn(
        "relative mx-auto w-full max-w-3xl px-3 pb-[max(1.5rem,env(safe-area-inset-bottom))] pt-2",
        className,
      )}
    >
      {notice && (
        <div
          ref={noticeRef}
          className="pointer-events-none absolute bottom-full left-0 right-0 z-20 mb-2.5 pb-2"
        >
          <div className="pointer-events-auto relative px-3">{notice}</div>
        </div>
      )}
      {children}
      {paused && <PausedDraftNote />}
    </motion.div>
  );
}
