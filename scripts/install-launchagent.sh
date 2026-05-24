#!/usr/bin/env bash
set -euo pipefail

LABEL="com.kkonstant.ytranslate"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RUNNER="${PROJECT_ROOT}/scripts/run-ytranslate-server.sh"
PLIST_PATH="${HOME}/Library/LaunchAgents/${LABEL}.plist"
LOG_DIR="${HOME}/Library/Logs/ytranslate"
PATH_VALUE="${HOME}/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
GUI_DOMAIN="gui/$(id -u)"

xml_escape() {
  python3 -c 'import html, sys; print(html.escape(sys.argv[1], quote=True))' "$1"
}

render_plist() {
  local escaped_runner escaped_project escaped_path escaped_stdout escaped_stderr
  escaped_runner="$(xml_escape "${RUNNER}")"
  escaped_project="$(xml_escape "${PROJECT_ROOT}")"
  escaped_path="$(xml_escape "${PATH_VALUE}")"
  escaped_stdout="$(xml_escape "${LOG_DIR}/server.log")"
  escaped_stderr="$(xml_escape "${LOG_DIR}/server.err.log")"

  cat <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>${LABEL}</string>
  <key>ProgramArguments</key>
  <array>
    <string>${escaped_runner}</string>
  </array>
  <key>WorkingDirectory</key>
  <string>${escaped_project}</string>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key>
    <string>${escaped_path}</string>
    <key>PYTHONUNBUFFERED</key>
    <string>1</string>
  </dict>
  <key>StandardOutPath</key>
  <string>${escaped_stdout}</string>
  <key>StandardErrorPath</key>
  <string>${escaped_stderr}</string>
</dict>
</plist>
PLIST
}

install_launchagent() {
  if [[ ! -x "${RUNNER}" ]]; then
    echo "Runner is not executable: ${RUNNER}" >&2
    exit 1
  fi

  mkdir -p "$(dirname "${PLIST_PATH}")" "${LOG_DIR}"
  render_plist > "${PLIST_PATH}"
  plutil -lint "${PLIST_PATH}" >/dev/null

  launchctl bootout "${GUI_DOMAIN}" "${PLIST_PATH}" >/dev/null 2>&1 || true
  launchctl bootstrap "${GUI_DOMAIN}" "${PLIST_PATH}"
  launchctl enable "${GUI_DOMAIN}/${LABEL}"
  launchctl kickstart -k "${GUI_DOMAIN}/${LABEL}"
  launchctl print "${GUI_DOMAIN}/${LABEL}" >/dev/null

  echo "Installed and started ${LABEL}"
  echo "Plist: ${PLIST_PATH}"
  echo "Logs: ${LOG_DIR}/server.log and ${LOG_DIR}/server.err.log"
}

uninstall_launchagent() {
  launchctl bootout "${GUI_DOMAIN}" "${PLIST_PATH}" >/dev/null 2>&1 || true
  rm -f "${PLIST_PATH}"
  echo "Uninstalled ${LABEL}"
}

status_launchagent() {
  launchctl print "${GUI_DOMAIN}/${LABEL}"
}

case "${1:-install}" in
  install|--install)
    install_launchagent
    ;;
  uninstall|--uninstall|remove|--remove)
    uninstall_launchagent
    ;;
  status|--status)
    status_launchagent
    ;;
  --print-plist)
    render_plist
    ;;
  *)
    echo "Usage: $0 [install|uninstall|status|--print-plist]" >&2
    exit 2
    ;;
esac
