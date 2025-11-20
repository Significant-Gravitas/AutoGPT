type Props = {
  branchName: string;
};

export function PreviewBanner({ branchName }: Props) {
  if (!branchName) {
    return null;
  }

  return (
    <div className="sticky top-0 z-50 w-full bg-green-500 px-4 py-2 text-center text-sm font-medium text-white">
      This is a Preview build for the branch:{" "}
      <span className="font-mono font-semibold">{branchName}</span>
    </div>
  );
}
