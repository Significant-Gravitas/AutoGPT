type Props = {
  children: React.ReactNode;
};

export function SelectedActionsWrap({ children }: Props) {
  return (
    <div className="my-0 ml-4 flex flex-row items-center gap-3 lg:mx-0 lg:my-4 lg:flex-col">
      {children}
    </div>
  );
}
