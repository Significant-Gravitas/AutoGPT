type Props = {
  code: string;
};

export function StoryCode(props: Props) {
  return (
    <pre className="block rounded border bg-zinc-100 px-3 py-2 font-mono text-xs text-indigo-800 shadow-sm">
      {props.code}
    </pre>
  );
}
