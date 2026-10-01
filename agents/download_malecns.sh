#!/bin/bash
# Downloads the three MaleCNS v1.0 connectome files nfly needs (public, CC-BY 4.0, no login).
# ~1.2 GB total: body-annotations (14 MB), body-neurotransmitters (43 MB),
# connectome-weights (1.1 GB). Source: https://male-cns.janelia.org/download/
set -euo pipefail

DATA_DIR="${1:-$(dirname "$0")/../data}"
mkdir -p "$DATA_DIR"
B=https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome

curl -L --fail -o "$DATA_DIR/body-annotations.feather" \
    "$B/body-annotations-male-cns-v1.0-minconf-0.5.feather"
curl -L --fail -o "$DATA_DIR/body-neurotransmitters.feather" \
    "$B/body-neurotransmitters-male-cns-v1.0.feather"
curl -L --fail -o "$DATA_DIR/connectome-weights.feather" \
    "$B/connectome-weights-male-cns-v1.0-minconf-0.5.feather"

echo "done: $DATA_DIR"
