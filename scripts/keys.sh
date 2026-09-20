#!/usr/bin/env bash
# Manages the gev API's bearer keys: the comma-separated list in Secret Manager that the API reads at
# instance start. Writes a new secret version, then rolls the API services so they pick it up. Only the
# small Node services restart; the model server is not touched.
#
#   ./scripts/keys.sh add alice       # issues sk-gev-alice-…, printed once
#   ./scripts/keys.sh revoke alice
#   ./scripts/keys.sh list            # names only, never the keys
set -euo pipefail

# Not the gcloud default: the owner's default project is a different one.
PROJECT="${GOOGLE_CLOUD_PROJECT:-gev-systemone}"
REGION="${REGION:-us-central1}"
SECRET="${SECRET:-gev-api-keys}"
# Every API service that mounts the secret; the first one is used to check the result.
SERVICES="${SERVICES:-gev gev-scored}"

usage() {
  echo "usage: keys.sh add <name> | revoke <name> | list" >&2
  exit 1
}

COMMAND="${1:-}"
NAME="${2:-}"
case "$COMMAND" in
  add | revoke) [[ "$NAME" =~ ^[a-z0-9][a-z0-9-]*$ ]] || usage ;;
  list) ;;
  *) usage ;;
esac

IFS=',' read -r -a KEYS <<<"$(gcloud secrets versions access latest --secret "$SECRET" --project "$PROJECT")"

# A named key is sk-gev-<name>-<48 hex>; matching the whole shape keeps "bob" from matching "bob-test".
is_named() { [[ "$1" =~ ^sk-gev-${NAME}-[0-9a-f]{48}$ ]]; }

if [[ "$COMMAND" == list ]]; then
  for key in "${KEYS[@]}"; do
    [[ "$key" =~ ^sk-gev-(.+)-[0-9a-f]{48}$ ]] && echo "${BASH_REMATCH[1]}" || echo "(unnamed)"
  done
  exit 0
fi

# Fail before touching the secret, not halfway through the roll.
for service in $SERVICES; do
  gcloud run services describe "$service" --project "$PROJECT" --region "$REGION" --format 'value(metadata.name)' >/dev/null
done

# Writes the list as a new secret version and restarts the APIs on it.
publish() {
  local joined version service
  joined="$(IFS=','; printf '%s' "$*")"
  version="$(printf '%s' "$joined" | gcloud secrets versions add "$SECRET" --project "$PROJECT" --data-file=- --format 'value(name)')"
  version="${version##*/}"
  for service in $SERVICES; do
    gcloud run services update "$service" --project "$PROJECT" --region "$REGION" \
      --update-env-vars "GEV_KEYS_VERSION=${version}" >/dev/null
    echo "Rolled ${service} onto version ${version} of ${SECRET}"
  done
}

# The status of an empty request made with a key: 422 means the key was accepted, 401 that it wasn't.
# The header goes through stdin so the key never shows up in a process list.
probe() {
  local url
  url="$(gcloud run services describe "${SERVICES%% *}" --project "$PROJECT" --region "$REGION" --format 'value(status.url)')"
  curl -s -m 30 -o /dev/null -w '%{http_code}' "${url}/v1/systemone" -H @- -H 'content-type: application/json' -d '{}' \
    <<<"authorization: Bearer $1" || true
}

if [[ "$COMMAND" == add ]]; then
  for key in "${KEYS[@]}"; do
    if is_named "$key"; then
      echo "There is already a key named ${NAME}. Revoke it first, or pick another name." >&2
      exit 1
    fi
  done
  KEY="sk-gev-${NAME}-$(openssl rand -hex 24)"
  publish "${KEYS[@]}" "$KEY"
  STATUS="$(probe "$KEY")"
  [[ "$STATUS" == 422 ]] && echo "Checked: ${SERVICES%% *} accepts the key." ||
    echo "Warning: ${SERVICES%% *} answered ${STATUS} to the new key (expected 422). Try again in a minute." >&2
  echo "API key for ${NAME} (shown once): ${KEY}"
else
  KEPT=()
  REVOKED=""
  for key in "${KEYS[@]}"; do
    if is_named "$key"; then REVOKED="$key"; else KEPT+=("$key"); fi
  done
  if [[ -z "$REVOKED" ]]; then
    echo "No key named ${NAME}. ./scripts/keys.sh list shows the names." >&2
    exit 1
  fi
  # An empty key list means an open API (see apiKeysFromEnv), so never publish one.
  if [[ ${#KEPT[@]} -eq 0 ]]; then
    echo "Refusing to revoke the last key: an empty list would leave the API open." >&2
    exit 1
  fi
  publish "${KEPT[@]}"
  STATUS="$(probe "$REVOKED")"
  [[ "$STATUS" == 401 ]] && echo "Checked: ${SERVICES%% *} now rejects ${NAME}'s key." ||
    echo "Warning: ${SERVICES%% *} answered ${STATUS} to the revoked key (expected 401). Try again in a minute." >&2
fi
