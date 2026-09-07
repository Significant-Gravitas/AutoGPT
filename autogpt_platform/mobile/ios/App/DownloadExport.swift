import AutoGPTMobileCore
import UIKit
import WebKit

@MainActor
final class DownloadExport: NSObject, WKDownloadDelegate {
  private static let maximumBytes: Int64 = 50 * 1024 * 1024
  private weak var presenter: UIViewController?
  private weak var shareController: UIActivityViewController?
  private let origin: AppOrigin
  private let completion: () -> Void
  private let onError: (String) -> Void
  private var activeDownload: WKDownload?
  private var directory: URL?
  private var file: URL?
  private var progressObservation: NSKeyValueObservation?
  private var isFinished = false

  init(
    presenter: UIViewController, origin: AppOrigin, download: WKDownload,
    completion: @escaping () -> Void, onError: @escaping (String) -> Void
  ) {
    self.presenter = presenter
    self.origin = origin
    activeDownload = download
    self.completion = completion
    self.onError = onError
  }

  func cancel() {
    finish(cancelDownload: true)
  }

  func download(
    _ download: WKDownload, decideDestinationUsing response: URLResponse,
    suggestedFilename: String, completionHandler: @escaping @MainActor @Sendable (URL?) -> Void
  ) {
    guard !isFinished, activeDownload === download else {
      completionHandler(nil)
      return
    }
    guard let url = response.url, origin.contains(url) || origin.allowsBlobDownload(url) else {
      completionHandler(nil)
      finish(message: "This download left your AutoGPT server. Open the file in your browser.")
      return
    }
    guard response.expectedContentLength <= Self.maximumBytes else {
      completionHandler(nil)
      finish(message: "Files larger than 50 MiB must be downloaded in your browser.")
      return
    }
    do {
      let exports = FileManager.default.temporaryDirectory.appendingPathComponent(
        "AutoGPTExports", isDirectory: true)
      try FileManager.default.createDirectory(at: exports, withIntermediateDirectories: true)
      let directory = exports.appendingPathComponent(UUID().uuidString, isDirectory: true)
      try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
      self.directory = directory
      let destination = directory.appendingPathComponent(Self.safeFilename(suggestedFilename))
      file = destination
      progressObservation = download.progress.observe(\.completedUnitCount, options: [.new]) {
        [weak self] progress, _ in
        let bytes = progress.completedUnitCount
        Task { @MainActor in
          guard let self, !self.isFinished, bytes > Self.maximumBytes else { return }
          self.finish(
            message: "Files larger than 50 MiB must be downloaded in your browser.",
            cancelDownload: true)
        }
      }
      completionHandler(destination)
    } catch {
      completionHandler(nil)
      finish(
        message: "There isn't enough space to prepare this download. Free some space and retry.")
    }
  }

  func downloadDidFinish(_ download: WKDownload) {
    guard !isFinished, activeDownload === download else { return }
    activeDownload = nil
    progressObservation = nil
    guard let file else {
      finish(message: "The downloaded file could not be found. Please try again.")
      return
    }
    do {
      let attributes = try file.resourceValues(forKeys: [.fileSizeKey, .isRegularFileKey])
      guard attributes.isRegularFile == true, let size = attributes.fileSize else {
        finish(message: "The downloaded file could not be opened. Please try again.")
        return
      }
      guard Int64(size) <= Self.maximumBytes else {
        finish(message: "Files larger than 50 MiB must be downloaded in your browser.")
        return
      }
    } catch {
      finish(message: "The downloaded file could not be opened. Please try again.")
      return
    }
    guard let presenter, presenter.viewIfLoaded?.window != nil,
      UIApplication.shared.applicationState == .active, !presenter.isBeingDismissed
    else {
      finish(message: "Return to AutoGPT and download the file again to save or share it.")
      return
    }
    guard presenter.presentedViewController == nil,
      presenter.navigationController?.presentedViewController == nil,
      !presenter.isBeingPresented
    else {
      finish(message: "Close the open dialog, then download this file again to save or share it.")
      return
    }
    let share = UIActivityViewController(activityItems: [file], applicationActivities: nil)
    shareController = share
    share.popoverPresentationController?.sourceView = presenter.view
    share.popoverPresentationController?.sourceRect = CGRect(
      x: presenter.view.bounds.midX, y: presenter.view.bounds.midY, width: 1, height: 1)
    share.completionWithItemsHandler = { [weak self] _, _, _, error in
      self?.finish(message: error == nil ? nil : "The file could not be shared. Please try again.")
    }
    presenter.present(share, animated: true)
    Task { @MainActor [weak self, share] in
      await Task.yield()
      guard let self, !self.isFinished,
        share.presentingViewController == nil, !share.isBeingPresented
      else { return }
      self.finish(message: "The save sheet could not open. Close any open dialog and try again.")
    }
  }

  func download(_ download: WKDownload, didFailWithError error: Error, resumeData: Data?) {
    guard !isFinished, activeDownload === download else { return }
    finish(message: "The download stopped before it finished. Check your connection and try again.")
  }

  func download(
    _ download: WKDownload, willPerformHTTPRedirection response: HTTPURLResponse,
    newRequest request: URLRequest,
    decisionHandler: @escaping @MainActor @Sendable (WKDownload.RedirectPolicy) -> Void
  ) {
    guard !isFinished, activeDownload === download else {
      decisionHandler(.cancel)
      return
    }
    guard request.url.map(origin.contains) == true else {
      decisionHandler(.cancel)
      finish(
        message: "This download left your AutoGPT server. Open the file in your browser.",
        cancelDownload: true)
      return
    }
    decisionHandler(.allow)
  }

  private func finish(message: String? = nil, cancelDownload: Bool = false) {
    guard !isFinished else { return }
    isFinished = true
    progressObservation = nil
    let download = activeDownload
    activeDownload = nil
    let share = shareController
    share?.completionWithItemsHandler = nil
    shareController = nil
    Task { @MainActor [self] in
      if cancelDownload, let download { _ = await download.cancel() }
      if let share, share.presentingViewController != nil, !share.isBeingDismissed {
        await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
          share.dismiss(animated: false) { continuation.resume() }
        }
      }
      if let directory { try? FileManager.default.removeItem(at: directory) }
      directory = nil
      file = nil
      completion()
      if let message { onError(message) }
    }
  }

  private static func safeFilename(_ suggested: String) -> String {
    let allowed = CharacterSet.alphanumerics.union(CharacterSet(charactersIn: " ._-"))
    let normalized = String(suggested.prefix(256)).precomposedStringWithCanonicalMapping
    let sanitized = normalized.unicodeScalars.map { allowed.contains($0) ? String($0) : "_" }
      .joined().trimmingCharacters(in: CharacterSet(charactersIn: " ."))
    let candidate = sanitized.isEmpty ? "download" : sanitized
    let fileExtension = (candidate as NSString).pathExtension
    let suffix =
      !fileExtension.isEmpty && fileExtension.utf8.count <= 16 ? ".\(fileExtension)" : ""
    let base = suffix.isEmpty ? candidate : String(candidate.dropLast(suffix.count))
    var filename = ""
    for character in base {
      let next = String(character)
      if filename.utf8.count + next.utf8.count + suffix.utf8.count > 180 { break }
      filename.append(character)
    }
    return (filename.isEmpty ? "download" : filename) + suffix
  }
}
