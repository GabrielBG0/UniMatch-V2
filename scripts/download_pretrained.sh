#!/usr/bin/env bash
# Download DINOv2 pre-trained backbone weights from Meta's servers.
# Saves to /petrobr/parceirosbr/spfm/pt-weights/dinov2/ and symlinks ./pretrained/ there.
# Usage: sh scripts/download_pretrained.sh [small] [base] [large] [giant]
#   No args → downloads small, base, large (sufficient for all default configs)

set -euo pipefail

SHARED_DIR="/petrobr/parceirosbr/spfm/pt-weights/dinov2"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PRETRAINED_LINK="$REPO_DIR/pretrained"

BASE_URL="https://dl.fbaipublicfiles.com/dinov2"

declare -A URLS=(
    [small]="$BASE_URL/dinov2_vits14/dinov2_vits14_pretrain.pth"
    [base]="$BASE_URL/dinov2_vitb14/dinov2_vitb14_pretrain.pth"
    [large]="$BASE_URL/dinov2_vitl14/dinov2_vitl14_pretrain.pth"
    [giant]="$BASE_URL/dinov2_vitg14/dinov2_vitg14_pretrain.pth"
)

declare -A SIZES=(
    [small]="~86 MB"
    [base]="~330 MB"
    [large]="~1.1 GB"
    [giant]="~4.3 GB"
)

if [ $# -eq 0 ]; then
    TARGETS=(small base large)
else
    TARGETS=("$@")
fi

mkdir -p "$SHARED_DIR"

# Create ./pretrained -> shared dir symlink if not already pointing there
if [ -L "$PRETRAINED_LINK" ]; then
    CURRENT_TARGET="$(readlink "$PRETRAINED_LINK")"
    if [ "$CURRENT_TARGET" != "$SHARED_DIR" ]; then
        echo "Updating symlink: $PRETRAINED_LINK -> $SHARED_DIR (was $CURRENT_TARGET)"
        ln -sfn "$SHARED_DIR" "$PRETRAINED_LINK"
    fi
elif [ -d "$PRETRAINED_LINK" ]; then
    echo "Moving existing pretrained/ contents to $SHARED_DIR ..."
    mv "$PRETRAINED_LINK"/*.pth "$SHARED_DIR"/ 2>/dev/null || true
    rm -rf "$PRETRAINED_LINK"
    ln -s "$SHARED_DIR" "$PRETRAINED_LINK"
else
    ln -s "$SHARED_DIR" "$PRETRAINED_LINK"
fi

echo "pretrained/ -> $SHARED_DIR"
echo ""

for MODEL in "${TARGETS[@]}"; do
    if [ -z "${URLS[$MODEL]+x}" ]; then
        echo "ERROR: unknown model '$MODEL'. Valid: small base large giant"
        exit 1
    fi

    DEST="$SHARED_DIR/dinov2_${MODEL}.pth"
    URL="${URLS[$MODEL]}"
    SIZE="${SIZES[$MODEL]}"

    if [ -f "$DEST" ]; then
        echo "Already exists: $DEST"
        continue
    fi

    echo "Downloading dinov2_${MODEL} (${SIZE})..."
    echo "  $URL"
    echo "  -> $DEST"

    TMP="$DEST.tmp"
    if command -v wget &>/dev/null; then
        wget -q --show-progress -O "$TMP" "$URL"
    else
        curl -L --progress-bar -o "$TMP" "$URL"
    fi

    mv "$TMP" "$DEST"
    echo "  Done."
done

echo ""
echo "Weights in $SHARED_DIR:"
ls -lh "$SHARED_DIR"/*.pth 2>/dev/null || echo "  (none yet)"
