// Suppresses expected act(...) warnings from React Query and component async updates.
// These warnings are normal behavior with React Query and don't indicate test failures.
export function suppressReactQueryUpdateWarning() {
    const originalError = console.error;
  
    console.error = (...args: unknown[]) => {
      const isActWarning = args.some(
        (arg) =>
          typeof arg === "string" &&
          (arg.includes("not wrapped in act(...)") ||
            arg.includes("An update to") && arg.includes("inside a test"))
      );
  
      if (isActWarning) {
        const fullMessage = args
          .map((arg) => String(arg))
          .join("\n")
          .toLowerCase();
  
        const isReactQueryRelated =
          fullMessage.includes("queryclientprovider") ||
          fullMessage.includes("react query") ||
          fullMessage.includes("@tanstack/react-query");
  
        if (isReactQueryRelated || fullMessage.includes("testproviders")) {
          return;
        }
      }
  
      originalError(...args);
    };
  
    // Return cleanup function
    return () => {
      console.error = originalError;
    };
  }