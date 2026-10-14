import Foundation

enum ConnectionResult: Sendable {
  /// The body ended. `entries` counts stored entries; zero means a synthetic-only response.
  case ended(entries: Int, syntheticError: String?)
  /// The route answered with an error status before streaming.
  case refused(HTTPError)
  /// Transport failure, liveness timeout, rotation or cancellation.
  case failed(any Error)
}

/// One SSE connection of a turn. Frames are handed over as they arrive; the
/// caller owns the turn state and decides whether and where to resume.
/// Cancelling the calling task ends the connection and releases its request.
enum TurnConnection {
  @MainActor
  static func run(
    open: @Sendable () async throws -> TurnOpen,
    onAccepted: () -> Void,
    onEntry: (StreamEntry) -> Void
  ) async -> ConnectionResult {
    let opened: TurnOpen
    do {
      opened = try await open()
    } catch {
      return .failed(error)
    }
    switch opened {
    case .refused(let error):
      return .refused(error)
    case .accepted(let frames):
      onAccepted()
      var entries = 0
      var syntheticError: String?
      do {
        for try await frame in frames {
          switch frame {
          case .entry(let turn, let entryID, let chunk):
            entries += 1
            onEntry(StreamEntry(turn: turn, entryID: entryID, chunk: chunk))
          case .synthetic(let chunk) where chunk.type == "error":
            syntheticError = chunk["errorText"]?.string ?? "error"
          default:
            break
          }
        }
      } catch {
        return .failed(error)
      }
      if Task.isCancelled { return .failed(CancellationError()) }
      return .ended(entries: entries, syntheticError: syntheticError)
    }
  }
}
