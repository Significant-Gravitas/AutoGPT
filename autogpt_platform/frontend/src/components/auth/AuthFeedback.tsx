interface Props {
  message?: string | null;
  isError?: boolean;
}

export default function AuthFeedback({ message = "", isError = false }: Props) {
  return (
    <div className="mt-4 text-center text-sm font-medium leading-normal">
      {isError ? (
        <div className="text-red-500">{message}</div>
      ) : (
        <div className="text-slate-950">{message}</div>
      )}
    </div>
  );
}
