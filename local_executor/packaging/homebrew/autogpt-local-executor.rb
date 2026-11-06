# Homebrew formula stub for autogpt-local-executor.
#
# Drop this into a tap (e.g. `Significant-Gravitas/homebrew-tap`) under
# `Formula/autogpt-local-executor.rb`. Users then install with:
#
#   brew install Significant-Gravitas/tap/autogpt-local-executor
#
# Post-PyPI-publish checklist (every tagged release;
# `scripts/update_packaging.sh` does steps 1-3):
#   1. Bump `version` to the new tag.
#   2. Replace `url` with the sdist URL on PyPI
#      (https://files.pythonhosted.org/packages/.../autogpt_local_executor-<v>.tar.gz).
#   3. Replace `sha256` with the sdist's sha256
#      (`shasum -a 256` of the downloaded sdist).
#   4. Required before this file goes to the tap: generate the `resource`
#      blocks with `brew update-python-resources` (see the note above
#      `def install`).
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

  # Every Python dependency needs a `resource` block here. Homebrew's pip
  # install runs with `--no-deps`, so it fetches nothing else. With no
  # resource blocks, pydantic, websockets, httpx, keyring and the rest are
  # missing, and `autogpt-shim` fails on its first import (the `test do`
  # block catches this). `scripts/update_packaging.sh` warns while there
  # are none.
  #
  # Generate them on every release, after `scripts/update_packaging.sh` has
  # set the PyPI `url`:
  #
  #   brew update-python-resources packaging/homebrew/autogpt-local-executor.rb
  #
  # Homebrew builds resources from source, and pydantic-core needs Rust to
  # build, so add `depends_on "rust" => :build` along with them.

  def install
    # Without resource blocks the install would "succeed" and leave a shim
    # that can't import its dependencies, so stop here instead.
    odie "No resource blocks. Run `brew update-python-resources` on this formula first." if resources.empty?

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
    assert_match "autogpt-shim #{version}", shell_output("#{bin}/autogpt-shim --version")
  end
end
