export default function AdminLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="flex min-h-screen w-full flex-col">
      <div className="flex-1 px-4">{children}</div>
    </div>
  );
}
