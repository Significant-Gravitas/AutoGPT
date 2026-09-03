interface Props {
  children: React.ReactNode;
}

export function ExpertSectionLabel({ children }: Props) {
  return (
    <div className="mb-3 text-xs font-medium uppercase tracking-[0.14em] text-zinc-400">
      {children}
    </div>
  );
}
