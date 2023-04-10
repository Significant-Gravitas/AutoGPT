const error = (error: unknown) => {
  console.error(error)
}

const errorHandler =
  (fn: Function) =>
  (...args: unknown[]) => {
    try {
      return fn(...args)
    } catch (err) {
      error(err)
    }
  }

export default {
  error,
  errorHandler,
}
