export function getInputPlaceholder(width?: number) {
  if (!width) return "What's your role and what eats up most of your day?";

  if (width < 500) {
    return "I'm a chef and I hate...";
  }
  if (width <= 1080) {
    return "What's your role and what eats up most of your day?";
  }
  return "What's your role and what eats up most of your day? e.g. 'I'm a recruiter and I hate...'";
}
