interface Props {
  label: string;
  value: string;
  alert?: boolean;
}

export function Summary({ label, value, alert = false }: Props) {
  return (
    <article className={alert ? "summary alert" : "summary"}>
      <span>{label}</span>
      <strong>{value}</strong>
    </article>
  );
}
