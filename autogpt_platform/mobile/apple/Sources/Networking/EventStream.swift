import Foundation

/// One JSON chunk of the AI SDK UI message stream, as the backend writes it.
struct WireChunk: Sendable, Equatable {
  let type: String
  let fields: [String: JSONValue]

  subscript(key: String) -> JSONValue? { fields[key] }
}

/// An `entry` is a stored stream entry with its `<turn>:<entry>` id and can be
/// resumed after. A `synthetic` frame has no id because the route wrote it
/// itself (an empty queued response or a route error). Comments are the
/// liveness signal: heartbeats and usage arrive as SSE comments.
enum SSEFrame: Sendable, Equatable {
  case entry(turn: String, entryID: String, chunk: WireChunk)
  case synthetic(WireChunk)
  case comment(String)
  case done

  init?(_ event: SSEParser.Event) {
    switch event {
    case .comment(let text):
      self = .comment(text)
    case .message(let id, let data):
      if data == "[DONE]" {
        self = .done
        return
      }
      guard let object = JSONValue.parse(Data(data.utf8))?.object,
        let type = object["type"]?.string
      else { return nil }
      let chunk = WireChunk(type: type, fields: object)
      if let id, let parsed = Self.parseID(id) {
        self = .entry(turn: parsed.turn, entryID: parsed.entryID, chunk: chunk)
      } else {
        self = .synthetic(chunk)
      }
    }
  }

  /// `<turn>:<entry>`: an entry id never holds a colon, a turn id might.
  static func parseID(_ id: String) -> (turn: String, entryID: String)? {
    guard let split = id.lastIndex(of: ":"), split > id.startIndex,
      id.index(after: split) < id.endIndex
    else { return nil }
    return (String(id[..<split]), String(id[id.index(after: split)...]))
  }
}

/// A minimal WHATWG event-stream line parser over raw bytes. Lines are only
/// decoded once complete, so a UTF-8 sequence split across network chunks
/// stays intact. Ids never carry over between events.
struct SSEParser: Sendable {
  enum Event: Equatable, Sendable {
    case message(id: String?, data: String)
    case comment(String)
  }

  private var line: [UInt8] = []
  private var data: [String] = []
  private var id: String?
  private var pendingCR = false

  mutating func feed(_ bytes: Data) -> [Event] {
    var events: [Event] = []
    for byte in bytes {
      if pendingCR {
        pendingCR = false
        if byte == 0x0A { continue }
      }
      if byte == 0x0A || byte == 0x0D {
        pendingCR = byte == 0x0D
        if let event = take(String(decoding: line, as: UTF8.self)) { events.append(event) }
        line.removeAll(keepingCapacity: true)
      } else {
        line.append(byte)
      }
    }
    return events
  }

  private mutating func take(_ line: String) -> Event? {
    if line.isEmpty {
      defer {
        data = []
        id = nil
      }
      return data.isEmpty ? nil : .message(id: id, data: data.joined(separator: "\n"))
    }
    if line.hasPrefix(":") {
      return .comment(String(line.dropFirst()).trimmed)
    }
    let field: Substring
    var value: Substring
    if let colon = line.firstIndex(of: ":") {
      field = line[..<colon]
      value = line[line.index(after: colon)...]
      if value.hasPrefix(" ") { value = value.dropFirst() }
    } else {
      field = line[...]
      value = ""
    }
    if field == "data" {
      data.append(String(value))
    } else if field == "id", !value.contains("\u{0}") {
      id = String(value)
    }
    return nil
  }
}

enum EventStreamResponse: Sendable {
  case open(HTTPHead, AsyncThrowingStream<SSEFrame, any Error>)
  /// A non-success status with up to 64 KB of its body.
  case refused(HTTPHead, Data)
}

/// No byte for 30 s (routes heartbeat every 10 s), or the 25 minute rotation
/// before the load balancer's 30 minute cut. Either way the caller resumes.
struct StreamInterrupted: Error {}

/// One streamed request, driven by URLSession's data delegate so the body
/// arrives in chunks and the request is cancelled as soon as nobody reads it.
final class EventStreamTask: NSObject, URLSessionDataDelegate, @unchecked Sendable {
  private static let maxRefusalBody = 64 * 1024
  private static let liveness: TimeInterval = 30
  private static let rotateAfter: TimeInterval = 25 * 60

  private let lock = NSLock()
  private var opening: CheckedContinuation<EventStreamResponse, any Error>?
  private var frames: AsyncThrowingStream<SSEFrame, any Error>.Continuation?
  private var stream: AsyncThrowingStream<SSEFrame, any Error>?
  private var dataTask: URLSessionDataTask?
  private var parser = SSEParser()
  private var head: HTTPHead?
  private var refusalBody = Data()
  private var lastActivity = Date()
  private var openedAt = Date()
  private var timers: [Task<Void, Never>] = []
  private var cancelled = false
  private var deadlinePassed = false

  func open(_ request: URLRequest, session: URLSession, headerTimeout: Duration) async throws
    -> EventStreamResponse
  {
    let (stream, continuation) = AsyncThrowingStream<SSEFrame, any Error>.makeStream()
    continuation.onTermination = { [weak self] _ in self?.stop() }
    lock.withLock {
      self.stream = stream
      self.frames = continuation
    }
    return try await withTaskCancellationHandler {
      try await withCheckedThrowingContinuation { opening in
        let task = session.dataTask(with: request)
        task.delegate = self
        let alreadyCancelled = lock.withLock {
          self.opening = opening
          self.dataTask = task
          return self.cancelled
        }
        if alreadyCancelled {
          finishOpening(throwing: TransportError.cancelled)
          return
        }
        addTimer {
          try? await Task.sleep(for: headerTimeout)
          $0.headerDeadlinePassed()
        }
        task.resume()
      }
    } onCancel: {
      self.cancelOpening()
    }
  }

  // MARK: URLSessionDataDelegate

  func urlSession(
    _ session: URLSession,
    task: URLSessionTask,
    willPerformHTTPRedirection response: HTTPURLResponse,
    newRequest request: URLRequest
  ) async -> URLRequest? {
    nil
  }

  func urlSession(
    _ session: URLSession,
    dataTask: URLSessionDataTask,
    didReceive response: URLResponse
  ) async -> URLSession.ResponseDisposition {
    guard let http = response as? HTTPURLResponse else {
      finishOpening(throwing: TransportError.offline(timedOut: false))
      return .cancel
    }
    let head = HTTPHead(http)
    if head.isRedirect {
      finishOpening(throwing: TransportError.redirectRefused)
      return .cancel
    }
    lock.withLock {
      self.head = head
      lastActivity = Date()
      openedAt = Date()
    }
    if head.isSuccess, let stream = lock.withLock({ self.stream }) {
      cancelTimers()
      startWatchdog()
      finishOpening(returning: .open(head, stream))
    }
    return .allow
  }

  func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
    let (frames, events) = lock.withLock {
      () -> (AsyncThrowingStream<SSEFrame, any Error>.Continuation?, [SSEParser.Event]) in
      lastActivity = Date()
      guard let head else { return (nil, []) }
      if head.isSuccess { return (self.frames, parser.feed(data)) }
      let room = Self.maxRefusalBody - refusalBody.count
      if room > 0 { refusalBody.append(data.prefix(room)) }
      return (nil, [])
    }
    guard let frames else { return }
    for event in events {
      if let frame = SSEFrame(event) { frames.yield(frame) }
    }
  }

  func urlSession(
    _ session: URLSession, task: URLSessionTask, didCompleteWithError error: (any Error)?
  ) {
    cancelTimers()
    let (head, body, frames, timedOut) = lock.withLock {
      (self.head, refusalBody, self.frames, deadlinePassed)
    }
    if let head, !head.isSuccess, error == nil {
      finishOpening(returning: .refused(head, body))
    } else if let error {
      let mapped = timedOut ? TransportError.offline(timedOut: true) : TransportError.from(error)
      finishOpening(throwing: mapped)
      frames?.finish(throwing: mapped)
      return
    } else {
      finishOpening(throwing: TransportError.offline(timedOut: false))
    }
    frames?.finish()
  }

  // MARK: Lifecycle

  private func finishOpening(returning response: EventStreamResponse) {
    lock.withLock { () -> CheckedContinuation<EventStreamResponse, any Error>? in
      defer { opening = nil }
      return opening
    }?.resume(returning: response)
  }

  private func finishOpening(throwing error: any Error) {
    let continuation = lock.withLock { () -> CheckedContinuation<EventStreamResponse, any Error>? in
      defer { opening = nil }
      return opening
    }
    guard let continuation else { return }
    continuation.resume(throwing: error)
    stop()
  }

  private func cancelOpening() {
    lock.withLock { cancelled = true }
    finishOpening(throwing: TransportError.cancelled)
  }

  private func headerDeadlinePassed() {
    let waiting = lock.withLock { () -> Bool in
      deadlinePassed = true
      return opening != nil
    }
    if waiting { finishOpening(throwing: TransportError.offline(timedOut: true)) }
  }

  /// The reader went away or the stream ended: release the request.
  private func stop() {
    cancelTimers()
    lock.withLock { dataTask }?.cancel()
  }

  private func startWatchdog() {
    addTimer { owner in
      while !Task.isCancelled {
        try? await Task.sleep(for: .seconds(5))
        if Task.isCancelled { return }
        if owner.isStale() {
          owner.interrupt()
          return
        }
      }
    }
  }

  private func isStale() -> Bool {
    lock.withLock {
      let now = Date()
      return now.timeIntervalSince(lastActivity) > Self.liveness
        || now.timeIntervalSince(openedAt) > Self.rotateAfter
    }
  }

  private func interrupt() {
    let (task, frames) = lock.withLock { (dataTask, self.frames) }
    frames?.finish(throwing: StreamInterrupted())
    task?.cancel()
  }

  private func addTimer(_ body: @escaping @Sendable (EventStreamTask) async -> Void) {
    let timer = Task { [weak self] in
      guard let self else { return }
      await body(self)
    }
    lock.withLock { timers.append(timer) }
  }

  private func cancelTimers() {
    let running = lock.withLock { () -> [Task<Void, Never>] in
      defer { timers = [] }
      return timers
    }
    for timer in running { timer.cancel() }
  }
}
