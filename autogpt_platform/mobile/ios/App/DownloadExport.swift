import AutoGPTMobileCore
import UIKit
import WebKit

@MainActor
final class DownloadExport: NSObject, WKDownloadDelegate {
  private weak var presenter: UIViewController?
  private let origin: AppOrigin
  private let completion: () -> Void
  private var directory: URL?
  private var file: URL?
  private var progressObservation: NSKeyValueObservation?
  private var isFinished = false

  init(presenter: UIViewController, origin: AppOrigin, completion: @escaping () -> Void) {
    self.presenter = presenter
    self.origin = origin
    self.completion = completion
  }

  func download(
    _ download: WKDownload, decideDestinationUsing response: URLResponse,
    suggestedFilename: String, completionHandler: @escaping @MainActor @Sendable (URL?) -> Void
  ) {
    guard let url = response.url, origin.contains(url) || origin.allowsBlobDownload(url),
      response.expectedContentLength <= 100 * 1024 * 1024
    else {
      completionHandler(nil)
      completion()
      return
    }
    do {
      let directory = FileManager.default.temporaryDirectory.appendingPathComponent(
        UUID().uuidString)
      try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
      self.directory = directory
      let name = URL(fileURLWithPath: suggestedFilename).lastPathComponent
      let destination = directory.appendingPathComponent(
        name.isEmpty || name == "." ? "download" : name)
      file = destination
      progressObservation = download.progress.observe(\.completedUnitCount, options: [.new]) {
        [weak self, weak download] progress, _ in
        if progress.completedUnitCount > 100 * 1024 * 1024 {
          Task { @MainActor in
            download?.cancel { _ in }
            self?.cleanup()
          }
        }
      }
      completionHandler(destination)
    } catch {
      completionHandler(nil)
      cleanup()
    }
  }

  func downloadDidFinish(_ download: WKDownload) {
    guard let file, let presenter, presenter.presentedViewController == nil else {
      cleanup()
      return
    }
    let share = UIActivityViewController(activityItems: [file], applicationActivities: nil)
    share.popoverPresentationController?.sourceView = presenter.view
    share.popoverPresentationController?.sourceRect = CGRect(
      x: presenter.view.bounds.midX, y: presenter.view.bounds.midY, width: 1, height: 1)
    share.completionWithItemsHandler = { [weak self] _, _, _, _ in self?.cleanup() }
    presenter.present(share, animated: true)
  }

  func download(_ download: WKDownload, didFailWithError error: Error, resumeData: Data?) {
    cleanup()
  }

  func download(
    _ download: WKDownload, willPerformHTTPRedirection response: HTTPURLResponse,
    newRequest request: URLRequest,
    decisionHandler: @escaping @MainActor @Sendable (WKDownload.RedirectPolicy) -> Void
  ) {
    decisionHandler(request.url.map(origin.contains) == true ? .allow : .cancel)
  }

  private func cleanup() {
    guard !isFinished else { return }
    isFinished = true
    progressObservation = nil
    if let directory { try? FileManager.default.removeItem(at: directory) }
    completion()
  }
}
