import Foundation
import Observation
import SwiftUI

enum TurnPhase: Equatable, Sendable {
  case idle, sending, streaming, reconnecting, stopping, queued, offline
}

enum LoadState: Equatable, Sendable { case loading, ready, error }

struct ConversationSnapshot: Equatable {
  var sessionID: String?
  var expertID: String?
  var loadState: LoadState
  var loadError: String?
  var items: [TimelineItem]
  var hasOlder: Bool
  var loadingOlder: Bool
  var refreshing: Bool
  var turn: TurnPhase
  var statusText: String?
  var notice: String?
  /// Increments with every published change, for scroll-follow.
  var revision: Int

  static let empty = ConversationSnapshot(
    loadState: .loading, items: [], hasOlder: false, loadingOlder: false, refreshing: false,
    turn: .idle, revision: 0)

  var canSend: Bool { loadState == .ready && turn == .idle }
  var canStop: Bool { [.streaming, .reconnecting, .queued, .offline].contains(turn) }
}

/// Mutable state one connection attempt shares with its callbacks.
@MainActor
private final class TurnAttempt {
  var pendingSendID: String?
  var failures = 0

  init(pendingSendID: String?) {
    self.pendingSendID = pendingSendID
  }
}

/// Owns one conversation for the life of its screen: persisted history, at
/// most one running turn, and messages sent from this device.
///
/// Sends are never replayed automatically. A send that fails, or whose
/// delivery can't be confirmed, stays visible as failed until the user
/// retries it with the same idempotency key or discards it. Failed sends are
/// memory-only: the server's durable cancellation contract isn't complete,
/// so nothing is queued for replay after a relaunch.
@MainActor
@Observable
final class ConversationController {
  private static let pageSize = 50
  private static let maxPageSize = 200
  private static let backoff: [Duration] = [
    .seconds(1), .seconds(2), .seconds(4), .seconds(8), .seconds(16),
  ]
  private static let maxReconnects = 8
  private static let emptyResumePause: Duration = .seconds(2)
  private static let queuedPoll: Duration = .seconds(5)
  private static let queuedPolls = 60
  private static let stopSettle: Duration = .seconds(4)
  private static let emitThrottle: Duration = .milliseconds(50)
  private static let foregroundReconnectAfter: TimeInterval = 5

  private(set) var snapshot = ConversationSnapshot.empty

  private let service: any ConversationService
  @ObservationIgnored private var sessionID: String?
  @ObservationIgnored private var expertID: String?
  @ObservationIgnored private var history: [Row] = [] {
    didSet { historyRevision += 1 }
  }
  @ObservationIgnored private var historyRevision = 0
  @ObservationIgnored private var historyCache:
    (revision: Int, running: Bool, items: [TimelineItem])?
  @ObservationIgnored private var membership: TurnMembership?
  @ObservationIgnored private var oldestSequence: Int?
  @ObservationIgnored private var hasOlder = false
  @ObservationIgnored private var live: TurnLog?
  @ObservationIgnored private var replayedUserContents: [String] = []
  @ObservationIgnored private var locals: [LocalMessage] = []
  @ObservationIgnored private var loadState: LoadState
  @ObservationIgnored private var loadError: String?
  @ObservationIgnored private var loadingOlder = false
  @ObservationIgnored private var refreshing = false
  @ObservationIgnored private var turn: TurnPhase = .idle
  @ObservationIgnored private var notice: String?
  @ObservationIgnored private var epoch = 0
  @ObservationIgnored private var connection: Task<ConnectionResult, Never>?
  private let waker = Waker()
  @ObservationIgnored private var backgroundedAt: Date?
  @ObservationIgnored private var disposed = true
  @ObservationIgnored private var emitTask: Task<Void, Never>?
  @ObservationIgnored private var revision = 0

  init(service: any ConversationService, sessionID: String?, expertID: String?) {
    self.service = service
    self.sessionID = sessionID
    self.expertID = expertID
    self.loadState = sessionID == nil ? .ready : .loading
    snapshot = buildSnapshot()
  }

  /// Starts (or restarts after `dispose`) loading and following the conversation.
  func start() {
    guard disposed else { return }
    disposed = false
    if loadState == .ready {
      Task { await refresh() }
    } else {
      Task { await load() }
    }
  }

  /// Stops listening. The server keeps running any turn; reopening resumes it.
  func dispose() {
    disposed = true
    epoch += 1
    connection?.cancel()
    connection = nil
    waker.rouse()
    emitTask?.cancel()
    emitTask = nil
    if turn != .idle { turn = .offline }
    for local in locals where local.state == .sending {
      patchLocal(local.messageID) {
        $0.state = .failed
        $0.error = "Couldn't confirm this message was sent. Retry sends it once."
      }
    }
  }

  /// Retries a conversation that failed to open.
  func reload() {
    if loadState == .error { Task { await load() } }
  }

  func send(_ text: String) async {
    let message = text.trimmed
    guard !message.isEmpty, loadState == .ready, turn == .idle, !disposed else { return }
    let local = LocalMessage(
      messageID: UUID().uuidString.lowercased(), text: message, state: .sending, error: nil,
      createdAt: Date())
    locals.append(local)
    await deliver(local)
  }

  /// Re-sends a failed message with its original idempotency key.
  func retry(_ messageID: String) async {
    guard let local = locals.first(where: { $0.messageID == messageID && $0.state == .failed }),
      loadState == .ready, turn == .idle, !disposed
    else { return }
    patchLocal(messageID) {
      $0.state = .sending
      $0.error = nil
    }
    await deliver(local)
  }

  func discard(_ messageID: String) {
    locals.removeAll { $0.messageID == messageID && $0.state == .failed }
    emitNow()
  }

  func stop() async {
    guard let sessionID, [.streaming, .reconnecting, .queued, .offline].contains(turn) else {
      return
    }
    let previous = turn
    turn = .stopping
    emitNow()
    do {
      _ = try await service.cancelTurn(sessionID: sessionID)
    } catch {
      if disposed { return }
      turn = previous
      notice = "Couldn't stop: \(ChatErrors.describe(error))"
      emitNow()
      return
    }
    // Give the stream a moment to deliver the turn's finish, then settle from history.
    let epoch = self.epoch
    await waker.pause(Self.stopSettle)
    if disposed || turn != .stopping || epoch != self.epoch { return }
    await finishTurn(beginEpoch())
  }

  /// Pull to refresh, foregrounding, or a manual reconnect after going offline.
  func refresh() async {
    guard let sessionID, !refreshing, turn != .sending, turn != .stopping, !disposed else { return }
    if turn == .streaming || turn == .reconnecting {
      // Resume from the cursor straight away instead of waiting out a dead socket.
      connection?.cancel()
      waker.rouse()
      return
    }
    refreshing = true
    notice = nil
    emitNow()
    let epoch = beginEpoch()
    do {
      let detail = try await service.detail(
        sessionID: sessionID, limit: Self.pageSize, beforeSequence: nil)
      if isCurrent(epoch) { await adoptServerState(detail, epoch: epoch, finishedTurn: nil) }
    } catch {
      if isCurrent(epoch) { notice = ChatErrors.describe(error) }
    }
    refreshing = false
    if !disposed { emitNow() }
  }

  func loadOlder() async {
    guard let sessionID, hasOlder, !loadingOlder, let oldest = oldestSequence, !disposed else {
      return
    }
    loadingOlder = true
    emitNow()
    do {
      let detail = try await service.detail(
        sessionID: sessionID, limit: Self.pageSize, beforeSequence: oldest)
      if !disposed {
        history = Rows.merge(
          older: withoutTurnRows(Rows.fromPersisted(detail.messages)), newer: history)
        oldestSequence = detail.oldestSequence ?? oldestSequence
        hasOlder = detail.hasMore
      }
    } catch {
      if !disposed { notice = "Couldn't load earlier messages. \(ChatErrors.describe(error))" }
    }
    loadingOlder = false
    if !disposed { emitNow() }
  }

  func scenePhaseChanged(_ phase: ScenePhase) {
    switch phase {
    case .background:
      backgroundedAt = Date()
    case .active:
      guard !disposed else { return }
      let away = backgroundedAt.map { Date().timeIntervalSince($0) } ?? 0
      backgroundedAt = nil
      // A suspended app's socket is usually dead; resume from the cursor at once.
      if away > Self.foregroundReconnectAfter || turn == .offline { Task { await refresh() } }
    default:
      break
    }
  }

  // MARK: Loading

  private func load() async {
    guard let sessionID else { return }
    loadState = .loading
    loadError = nil
    emitNow()
    let epoch = beginEpoch()
    do {
      let detail = try await service.detail(
        sessionID: sessionID, limit: Self.pageSize, beforeSequence: nil)
      guard isCurrent(epoch) else { return }
      expertID = detail.expertID ?? expertID
      loadState = .ready
      await adoptServerState(detail, epoch: epoch, finishedTurn: nil)
    } catch {
      guard isCurrent(epoch) else { return }
      loadState = .error
      loadError = ChatErrors.describe(error)
      emitNow()
    }
  }

  /// Makes the server's view authoritative: history from its newest page,
  /// then whatever turn it reports as running or queued. `finishedTurn` is a
  /// turn this device just saw finish, which a lagging registry may still
  /// report as active.
  private func adoptServerState(_ detail: SessionDetail, epoch: Int, finishedTurn: String?) async {
    membership = nil
    live = nil
    replayedUserContents = []
    applyNewestPage(detail)
    dropAcceptedLocals()
    if let sessionID, let active = detail.activeStream, active.turnID != finishedTurn {
      if let checkpoint = active.checkpoint, detail.hasMore,
        (detail.oldestSequence ?? 0) > checkpoint.sequence
      {
        // The newest page starts inside the turn; fetch enough rows to seed it whole.
        let wider = try? await service.detail(
          sessionID: sessionID, limit: Self.maxPageSize, beforeSequence: nil)
        guard isCurrent(epoch) else { return }
        if let wider { applyNewestPage(wider) }
      }
      let split = ActiveTurnSplit(rows: history, active: active)
      membership = split.membership
      history = history.filter { !split.membership.contains($0) }
      live = split.seed
      replayedUserContents = split.replayedUserContents
      let service = self.service
      let turnID = active.turnID
      let after = split.after
      let next = beginEpoch()
      Task {
        await runTurn(
          epoch: next,
          open: {
            try await service.resumeTurn(sessionID: sessionID, turnID: turnID, after: after)
          },
          sendingID: nil)
      }
      return
    }
    if detail.chatStatus == "queued" {
      let next = beginEpoch()
      Task { await waitForQueuedTurn(epoch: next) }
      return
    }
    turn = .idle
    emitNow()
  }

  // MARK: Turns

  private func deliver(_ local: LocalMessage) async {
    turn = .sending
    notice = nil
    emitNow()
    var target = sessionID
    if target == nil {
      do {
        let created = try await service.create(expertID: expertID)
        target = created.id
        sessionID = created.id
        expertID = created.expertID ?? expertID
      } catch {
        failLocal(local.messageID, ChatErrors.describe(error))
        return
      }
    }
    guard let target, !disposed else {
      failLocal(local.messageID, "Couldn't confirm this message was sent. Retry sends it once.")
      return
    }
    live = TurnLog(status: .running)
    replayedUserContents = []
    let service = self.service
    let text = local.text
    let messageID = local.messageID
    await runTurn(
      epoch: beginEpoch(),
      open: { try await service.openTurn(sessionID: target, message: text, messageID: messageID) },
      sendingID: messageID)
  }

  private func runTurn(
    epoch: Int, open: @escaping @Sendable () async throws -> TurnOpen, sendingID: String?
  ) async {
    var opener = open
    let attempt = TurnAttempt(pendingSendID: sendingID)
    turn = sendingID != nil ? .sending : .streaming
    emitNow()
    while isCurrent(epoch) {
      let current = opener
      let task = Task { () -> ConnectionResult in
        await TurnConnection.run(
          open: current,
          onAccepted: { self.connectionAccepted(epoch: epoch, attempt: attempt) },
          onEntry: { self.connectionEntry($0, epoch: epoch, attempt: attempt) })
      }
      connection = task
      let result = await task.value
      guard isCurrent(epoch) else { return }
      if let pending = attempt.pendingSendID {
        await settleUnconfirmedSend(epoch: epoch, messageID: pending, result: result)
        return
      }
      if let log = live, log.status == .finished || log.status == .failed {
        await finishTurn(epoch)
        return
      }
      switch result {
      case .refused(let error):
        if await handleResumeRefusal(epoch: epoch, error: error) { return }
      case .ended(let entries, _) where entries == 0 || live?.turnID == nil:
        // Nothing (more) to stream: the send was queued or de-duplicated, or
        // the turn ended without a finish entry. Ask the server what happened.
        await reconcileFromServer(
          epoch: epoch, pause: live?.turnID != nil ? Self.emptyResumePause : .zero)
        return
      default:
        break
      }
      attempt.failures += 1
      if attempt.failures > Self.maxReconnects {
        turn = .offline
        notice = "Lost the connection to this conversation. It keeps running on the server."
        emitNow()
        return
      }
      // A Stop requested while the connection was down stays "stopping".
      if turn != .stopping { turn = .reconnecting }
      emitNow()
      await waker.pause(Self.backoff[min(attempt.failures, Self.backoff.count) - 1])
      guard isCurrent(epoch) else { return }
      guard let turnID = live?.turnID, let sessionID else {
        await reconcileFromServer(epoch: epoch, pause: .zero)
        return
      }
      let after = live?.cursor ?? "0-0"
      let service = self.service
      opener = { try await service.resumeTurn(sessionID: sessionID, turnID: turnID, after: after) }
    }
  }

  private func connectionAccepted(epoch: Int, attempt: TurnAttempt) {
    guard isCurrent(epoch) else { return }
    if let pending = attempt.pendingSendID { patchLocal(pending) { $0.state = .accepted } }
    attempt.pendingSendID = nil
    if turn != .stopping { turn = .streaming }
    emitNow()
  }

  private func connectionEntry(_ entry: StreamEntry, epoch: Int, attempt: TurnAttempt) {
    guard isCurrent(epoch) else { return }
    var log = live ?? TurnLog(status: .running)
    log.apply(entry)
    live = log
    attempt.failures = 0
    scheduleEmit()
  }

  /// 409: the cursor fell behind a trim, so reseed from its checkpoint.
  /// 410/404: the stream is gone. Returns true once the turn is settled.
  private func handleResumeRefusal(epoch: Int, error: HTTPError) async -> Bool {
    let checkpoint = TurnCheckpoint(error.body?["checkpoint"])
    if error.status == 409, let sessionID, let turnID = live?.turnID, let checkpoint {
      if let detail = try? await service.detail(
        sessionID: sessionID, limit: Self.maxPageSize, beforeSequence: nil)
      {
        guard isCurrent(epoch) else { return true }
        let rows = Rows.fromPersisted(detail.messages)
        let split = ActiveTurnSplit(
          rows: rows, active: ActiveStream(turnID: turnID, startedAt: nil, checkpoint: checkpoint))
        membership = split.membership
        history = Rows.merge(older: withoutTurnRows(history), newer: withoutTurnRows(rows))
        live = .seeded(turnID: turnID, rows: rows, checkpoint: checkpoint)
        replayedUserContents = []
      }
      // Otherwise retry the resume; the next refusal brings a fresh checkpoint.
      return false
    }
    if [409, 410, 404].contains(error.status) {
      await finishTurn(epoch)
      return true
    }
    return false
  }

  /// The send's response never arrived, so the server may or may not hold the
  /// message. If the newest user row is this message it was delivered;
  /// otherwise it shows as failed for a manual retry, which the server
  /// de-duplicates by the same idempotency key.
  private func settleUnconfirmedSend(epoch: Int, messageID: String, result: ConnectionResult) async
  {
    if case .refused(let error) = result {
      live = nil
      failLocal(messageID, ChatErrors.describe(error))
      return
    }
    if let sessionID, let local = locals.first(where: { $0.messageID == messageID }),
      let detail = try? await service.detail(
        sessionID: sessionID, limit: Self.pageSize, beforeSequence: nil)
    {
      guard isCurrent(epoch) else { return }
      let newestUser = Rows.fromPersisted(detail.messages).last { $0.role == .user }
      let delivered = newestUser.map { row in
        let sameText = Attachments.split(row.content).text == local.text
        guard let createdAt = row.createdAt else { return sameText }
        let recent =
          ISODate.parse(createdAt).map { $0 >= local.createdAt.addingTimeInterval(-60) } ?? false
        return sameText && recent
      }
      if delivered == true {
        locals.removeAll { $0.messageID == messageID }
        await adoptServerState(detail, epoch: epoch, finishedTurn: nil)
        return
      }
    }
    guard isCurrent(epoch) else { return }
    live = nil
    failLocal(messageID, "Couldn't confirm this message was sent. Retry sends it once.")
  }

  private func reconcileFromServer(epoch: Int, pause: Duration) async {
    guard let sessionID else { return }
    if pause > .zero {
      await waker.pause(pause)
      guard isCurrent(epoch) else { return }
    }
    do {
      let detail = try await service.detail(
        sessionID: sessionID, limit: Self.pageSize, beforeSequence: nil)
      if isCurrent(epoch) { await adoptServerState(detail, epoch: epoch, finishedTurn: nil) }
    } catch {
      guard isCurrent(epoch) else { return }
      turn = .offline
      notice = ChatErrors.describe(error)
      emitNow()
    }
  }

  private func waitForQueuedTurn(epoch: Int) async {
    turn = .queued
    emitNow()
    for _ in 0..<Self.queuedPolls {
      await waker.pause(Self.queuedPoll)
      guard isCurrent(epoch), let sessionID else { return }
      let detail = try? await service.detail(
        sessionID: sessionID, limit: Self.pageSize, beforeSequence: nil)
      guard isCurrent(epoch) else { return }
      if let detail, detail.activeStream != nil || detail.chatStatus != "queued" {
        await adoptServerState(detail, epoch: epoch, finishedTurn: nil)
        return
      }
    }
    guard isCurrent(epoch) else { return }
    turn = .idle
    notice = "Still waiting for a free slot to start this conversation's next reply."
    emitNow()
  }

  /// Replaces the live view with persisted history once the turn has ended.
  private func finishTurn(_ epoch: Int) async {
    guard let sessionID else { return }
    let finishedTurn = live?.turnID
    do {
      let detail = try await service.detail(
        sessionID: sessionID, limit: Self.pageSize, beforeSequence: nil)
      if isCurrent(epoch) {
        await adoptServerState(detail, epoch: epoch, finishedTurn: finishedTurn)
      }
    } catch {
      guard isCurrent(epoch) else { return }
      // Keep the finished turn on screen; history catches up on the next refresh.
      turn = .idle
      notice = "Couldn't refresh this conversation."
      emitNow()
    }
  }

  // MARK: History

  private func applyNewestPage(_ detail: SessionDetail) {
    let rows = withoutTurnRows(Rows.fromPersisted(detail.messages))
    // Older pages already loaded are kept; the newest page replaces its own range.
    let kept =
      detail.oldestSequence.map { oldest in
        history.filter { ($0.sequence ?? .max) < oldest }
      } ?? []
    history = Rows.merge(older: kept, newer: rows)
    if kept.isEmpty {
      oldestSequence = detail.oldestSequence
      hasOlder = detail.hasMore
    }
  }

  private func withoutTurnRows(_ rows: [Row]) -> [Row] {
    guard let membership else { return rows }
    return rows.filter { !membership.contains($0) }
  }

  private func dropAcceptedLocals() {
    locals.removeAll { $0.state == .accepted }
  }

  private func failLocal(_ messageID: String, _ error: String) {
    patchLocal(messageID) {
      $0.state = .failed
      $0.error = error
    }
    turn = .idle
    emitNow()
  }

  private func patchLocal(_ messageID: String, _ change: (inout LocalMessage) -> Void) {
    guard let index = locals.firstIndex(where: { $0.messageID == messageID }) else { return }
    change(&locals[index])
  }

  // MARK: Epochs and publishing

  private func beginEpoch() -> Int {
    epoch += 1
    connection?.cancel()
    connection = nil
    waker.rouse()
    return epoch
  }

  private func isCurrent(_ epoch: Int) -> Bool {
    !disposed && epoch == self.epoch
  }

  private func scheduleEmit() {
    guard emitTask == nil else { return }
    emitTask = Task { [weak self] in
      try? await Task.sleep(for: Self.emitThrottle)
      guard let self, !Task.isCancelled else { return }
      self.emitTask = nil
      self.emitNow()
    }
  }

  private func emitNow() {
    emitTask?.cancel()
    emitTask = nil
    snapshot = buildSnapshot()
  }

  private func buildSnapshot() -> ConversationSnapshot {
    let running = live?.status == .running
    let persisted: [TimelineItem]
    if let cache = historyCache, cache.revision == historyRevision, cache.running == running {
      persisted = cache.items
    } else {
      persisted = Timeline.historyItems(history, turnRunning: running)
      historyCache = (historyRevision, running, persisted)
    }
    var items = persisted
    items += locals.filter { $0.state != .failed }.map(Timeline.localItem)
    if let live { items += Timeline.liveItems(live, replayedUserContents: replayedUserContents) }
    items += locals.filter { $0.state == .failed }.map(Timeline.localItem)
    revision += 1
    return ConversationSnapshot(
      sessionID: sessionID, expertID: expertID, loadState: loadState, loadError: loadError,
      items: items, hasOlder: hasOlder, loadingOlder: loadingOlder, refreshing: refreshing,
      turn: turn, statusText: running ? live?.statusText : nil, notice: notice, revision: revision)
  }
}
