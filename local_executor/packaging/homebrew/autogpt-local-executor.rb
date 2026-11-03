# Homebrew formula stub for autogpt-local-executor.
#
# Drop this into a tap (e.g. `Significant-Gravitas/homebrew-tap`) under
# `Formula/autogpt-local-executor.rb`. Users then install with:
#
#   brew install Significant-Gravitas/tap/autogpt-local-executor
#
# Post-PyPI-publish checklist (CI should do this on every tagged release):
#   1. Bump `version` to the new tag.
#   2. Replace `url` with the sdist URL on PyPI
#      (https://files.pythonhosted.org/packages/.../autogpt_local_executor-<v>.tar.gz).
#   3. Replace `sha256` with the sdist's sha256
#      (`shasum -a 256` of the downloaded sdist).
#   4. Re-resolve `resource` blocks for pinned deps with
#      `brew update-python-resources autogpt-local-executor`.
#   5. Run `brew test-bot --tap=Significant-Gravitas/tap autogpt-local-executor`.
#
# For pre-PyPI testing, point `url` at the GitHub tarball:
#   url "https://github.com/Significant-Gravitas/autogpt-local-executor/archive/refs/tags/v0.0.1.tar.gz"

class AutogptLocalExecutor < Formula
  include Language::Python::Virtualenv

  desc "EXPERIMENTAL local PC shim for the AutoGPT hosted platform"
  homepage "https://github.com/Significant-Gravitas/autogpt-local-executor"
  url "PLACEHOLDER-https://files.pythonhosted.org/packages/source/a/autogpt-local-executor/autogpt-local-executor-0.0.1.tar.gz"
  sha256 "PLACEHOLDER-0000000000000000000000000000000000000000000000000000000000000000"
  license "MIT"
  version "0.0.1"

  depends_on "python@3.11"

  # `brew update-python-resources` will populate these once the package is on
  # PyPI. Until then, the install will fall back to fetching deps from PyPI
  # at install time, which Homebrew normally discourages but is acceptable
  # for an alpha tap.
  #
  # resource "pydantic" do
  #   url "..."
  #   sha256 "..."
  # end
  #
  # resource "websockets" do
  #   url "..."
  #   sha256 "..."
  # end
  #
  # ...

  def install
    virtualenv_install_with_resources
  end

  def caveats
    <<~EOS
      This is EXPERIMENTAL software that gives the AutoGPT hosted platform
      code execution access to this machine. Read docs/SECURITY.md before
      running it.

      One-time setup:
        autogpt-shim auth
        autogpt-shim install     # register the launchd LaunchAgent
        launchctl load ~/Library/LaunchAgents/net.autogpt.shim.plist

      Uninstall the autostart entry with:
        autogpt-shim uninstall
        launchctl unload ~/Library/LaunchAgents/net.autogpt.shim.plist
    EOS
  end

  test do
    assert_match "autogpt-shim", shell_output("#{bin}/autogpt-shim --help")
  end
end
