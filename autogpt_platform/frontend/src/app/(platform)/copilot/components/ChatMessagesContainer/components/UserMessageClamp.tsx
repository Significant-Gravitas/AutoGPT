import { useLayoutEffect, useRef, useState } from "react";

interface Props {
  children: React.ReactNode;
  trailing?: React.ReactNode;
}

export function UserMessageClamp({ children, trailing }: Props) {
  const contentRef = useRef<HTMLDivElement>(null);
  const [isOverflowing, setIsOverflowing] = useState(false);
  const [expanded, setExpanded] = useState(false);

  useLayoutEffect(() => {
    const el = contentRef.current;
    if (!el || expanded) return;
    setIsOverflowing(el.scrollHeight > el.clientHeight + 1);
  }, [expanded, children]);

  const showToggle = isOverflowing || expanded;

  return (
    <div>
      <div ref={contentRef} className={expanded ? undefined : "line-clamp-6"}>
        {children}
      </div>
      {(showToggle || trailing) && (
        <div className="mt-1 flex flex-wrap items-center gap-2">
          {showToggle && (
            <button
              type="button"
              className="text-xs font-medium text-purple-700 hover:underline"
              onClick={() => setExpanded(!expanded)}
            >
              {expanded ? "Show less" : "Read more"}
            </button>
          )}
          {trailing}
        </div>
      )}
    </div>
  );
}
