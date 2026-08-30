import { useLayoutEffect, useRef, useState } from "react";

interface Props {
  spec: string;
}

export function TaskSpec({ spec }: Props) {
  const specRef = useRef<HTMLParagraphElement>(null);
  const [isOverflowing, setIsOverflowing] = useState(false);
  const [expanded, setExpanded] = useState(false);

  useLayoutEffect(() => {
    const el = specRef.current;
    if (!el || expanded) return;
    setIsOverflowing(el.scrollHeight > el.clientHeight + 1);
  }, [expanded, spec]);

  return (
    <div className="mt-4">
      <p
        ref={specRef}
        data-testid="task-spec"
        className={`whitespace-pre-line text-[15px] leading-7 text-zinc-600 ${
          expanded ? "" : "line-clamp-6"
        }`}
      >
        {spec}
      </p>
      {isOverflowing || expanded ? (
        <button
          type="button"
          data-testid="task-spec-toggle"
          className="mt-2 text-sm font-medium text-zinc-900 hover:underline"
          onClick={() => setExpanded(!expanded)}
        >
          {expanded ? "Show less" : "Read more"}
        </button>
      ) : null}
    </div>
  );
}
