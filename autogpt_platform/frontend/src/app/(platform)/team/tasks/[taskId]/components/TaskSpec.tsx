import { useTaskSpec } from "./useTaskSpec";

interface Props {
  spec: string;
}

export function TaskSpec({ spec }: Props) {
  const { specRef, isOverflowing, expanded, toggleExpanded } =
    useTaskSpec(spec);

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
          onClick={toggleExpanded}
        >
          {expanded ? "Show less" : "Read more"}
        </button>
      ) : null}
    </div>
  );
}
