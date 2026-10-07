interface Props {
  code: string;
}

export function StoryCode(props: Props) {
  return (
    <pre className="block rounded-sm border bg-zinc-100 px-3 py-2 font-mono text-xs text-purple-800 shadow-xs">
      {props.code}
    </pre>
  );
}
