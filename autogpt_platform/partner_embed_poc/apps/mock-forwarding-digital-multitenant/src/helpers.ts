export function initials(name: string) {
  return name
    .split(" ")
    .map((part) => part[0])
    .join("")
    .slice(0, 2);
}

export function shortID(value: string) {
  return value.slice(0, 8) + "…" + value.slice(-4);
}
