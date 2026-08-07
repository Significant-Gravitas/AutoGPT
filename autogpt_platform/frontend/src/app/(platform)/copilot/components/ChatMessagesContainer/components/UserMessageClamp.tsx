import { useLayoutEffect, useRef, useState } from "react";

interface Props {
  children: React.ReactNode;
}

export function UserMessageClamp({ children }: Props) {
  const contentRef = useRef<HTMLDivElement>(null);
  const [isOverflowing, setIsOverflowing] = useState(false);
  const [expanded, setExpanded] = useState(false);

  useLayoutEffect(() => {
    const el = contentRef.current;
    if (!el || expanded) return;
    setIsOverflowing(el.scrollHeight > el.clientHeight + 1);
  }, [expanded, children]);

  return (
    <div>
      <div ref={contentRef} className={expanded ? undefined : "line-clamp-6"}>
        {children}
      </div>
      {(isOverflowing || expanded) && (
        <button
          type="button"
          className="mt-1 text-xs font-medium text-purple-700 hover:underline"
          onClick={() => setExpanded(!expanded)}
        >
          {expanded ? "Show less" : "Read more"}
        </button>
      )}
    </div>
  );
}
