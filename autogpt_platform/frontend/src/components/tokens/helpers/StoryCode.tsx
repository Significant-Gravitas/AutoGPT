type Props = {
  code: string;
};

export function StoryCode(props: Props) {
  return (
    <pre className="block rounded border bg-surface-hover px-3 py-2 font-mono text-xs text-foreground-emphasis shadow-sm">
      {props.code}
    </pre>
  );
}
