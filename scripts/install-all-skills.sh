#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/PredictiveScienceLab/codex-skills.git}"
REF="${REF:-main}"
DEST_ROOT="${DEST_ROOT:-${CODEX_HOME:-$HOME/.codex}/skills}"
FORCE="${FORCE:-0}"

usage() {
  cat <<'EOF'
Install all skills from this repository into $CODEX_HOME/skills.

Options:
  --repo <url>      Git repo URL (default: https://github.com/PredictiveScienceLab/codex-skills.git)
  --ref <ref>       Branch/tag/commit (default: main)
  --dest <path>     Destination root (default: ${CODEX_HOME:-$HOME/.codex}/skills)
  --force           Replace existing skill folders
  -h, --help        Show help

Examples:
  ./scripts/install-all-skills.sh
  ./scripts/install-all-skills.sh --force
  REPO_URL=git@github.com:PredictiveScienceLab/codex-skills.git ./scripts/install-all-skills.sh
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo)
      REPO_URL="$2"
      shift 2
      ;;
    --ref)
      REF="$2"
      shift 2
      ;;
    --dest)
      DEST_ROOT="$2"
      shift 2
      ;;
    --force)
      FORCE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

TMP_DIR="$(mktemp -d)"
cleanup() { rm -rf "$TMP_DIR"; }
trap cleanup EXIT

echo "Cloning $REPO_URL#$REF ..."
if ! git clone --depth 1 --branch "$REF" "$REPO_URL" "$TMP_DIR/repo"; then
  echo "Warning: could not clone ref '$REF'; falling back to repository default branch." >&2
  rm -rf "$TMP_DIR/repo"
  git clone --depth 1 "$REPO_URL" "$TMP_DIR/repo"
fi

mkdir -p "$DEST_ROOT"

installed=0
updated=0
skipped=0

for skill_dir in "$TMP_DIR/repo"/skills/*; do
  [[ -d "$skill_dir" ]] || continue
  skill_name="$(basename "$skill_dir")"
  dest="$DEST_ROOT/$skill_name"

  if [[ -e "$dest" ]]; then
    if [[ "$FORCE" == "1" ]]; then
      rm -rf "$dest"
      cp -R "$skill_dir" "$dest"
      updated=$((updated + 1))
      echo "Updated: $skill_name"
    else
      skipped=$((skipped + 1))
      echo "Skipped (exists): $skill_name"
    fi
  else
    cp -R "$skill_dir" "$dest"
    installed=$((installed + 1))
    echo "Installed: $skill_name"
  fi
done

echo
echo "Install complete."
echo "Installed: $installed"
echo "Updated:   $updated"
echo "Skipped:   $skipped"
echo "Restart Codex to pick up new skills."
