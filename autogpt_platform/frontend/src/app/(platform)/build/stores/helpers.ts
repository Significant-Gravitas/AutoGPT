export const accumulateExecutionData = (
  accumulated: Record<string, unknown[]>,
  data: Record<string, unknown> | undefined,
) => {
  if (!data) return { ...accumulated };
  const next = { ...accumulated };
  Object.entries(data).forEach(([key, values]) => {
    const nextValues = Array.isArray(values) ? values : [values];
    if (next[key]) {
      next[key] = [...next[key], ...nextValues];
    } else {
      next[key] = [...nextValues];
    }
  });
  return next;
};
