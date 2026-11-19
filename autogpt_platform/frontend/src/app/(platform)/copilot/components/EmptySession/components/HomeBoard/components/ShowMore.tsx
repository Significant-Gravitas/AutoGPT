import { Button } from "@/components/atoms/Button/Button";

interface Props {
  isLoading: boolean;
  onMore: () => void;
}

/** Reads the next page of tasks; they fill the lanes they belong in. */
export function ShowMore({ isLoading, onMore }: Props) {
  return (
    <div className="flex shrink-0 justify-center">
      <Button
        type="button"
        variant="secondary"
        size="xs"
        shape="pill"
        loading={isLoading}
        onClick={onMore}
      >
        Show more tasks
      </Button>
    </div>
  );
}
