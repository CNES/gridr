#!/usr/bin/env bash
SCRIPTPATH="$( cd -- "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
TEMPLATE_RUST=${SCRIPTPATH}/../templates/NOTICE_rust_3rdparty.hbs
CARGO_FILE=${SCRIPTPATH}/../rust/gridr/Cargo.toml
# Install cargo-about if not installed yet
cargo install cargo-about
echo "[rust-deps-licenses]"
cargo-about generate -m ${CARGO_FILE} ${TEMPLATE_RUST}
