import * as Haptics from "expo-haptics";

// A phone without a haptic engine (or a simulator) refuses; the tap is a
// nicety, so a refusal is dropped.
function ignore() {
  return undefined;
}

/** A message went out. */
export function tapSend() {
  void Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light).catch(ignore);
}

/** The first words of a reply arrived. */
export function tapReply() {
  void Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Soft).catch(ignore);
}

/** A reply was stopped. */
export function tapStop() {
  void Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium).catch(ignore);
}

/** A long press opened a menu. */
export function tapMenu() {
  void Haptics.selectionAsync().catch(ignore);
}
