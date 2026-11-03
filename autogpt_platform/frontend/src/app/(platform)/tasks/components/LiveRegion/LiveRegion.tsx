interface Props {
  message: string;
}

export function LiveRegion({ message }: Props) {
  return (
    <div className="sr-only" aria-live="polite" aria-atomic="true">
      {message}
    </div>
  );
}
