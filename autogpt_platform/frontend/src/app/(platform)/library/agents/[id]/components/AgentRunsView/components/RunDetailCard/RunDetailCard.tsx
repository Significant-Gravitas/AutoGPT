type Props = {
  children: React.ReactNode;
};

export function RunDetailCard({ children }: Props) {
  return (
    <div className="min-h-20 rounded-xlarge border border-slate-50/70 bg-white p-6">
      {children}
    </div>
  );
}
