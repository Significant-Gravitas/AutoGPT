export function isOpenUIEnabled() {
  return process.env.NEXT_PUBLIC_OPENUI_EXPERIMENT === "true";
}
