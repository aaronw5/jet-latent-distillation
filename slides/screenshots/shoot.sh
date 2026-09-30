#!/bin/bash
# Screenshots of a site page for the slides (headless Chrome; set CHROME to its binary if it is not in /Applications).
#   slides/screenshots/shoot.sh site/all/n8 out/        -> out/<step name>.png, plus the crops used on the slides
# The page loads its explanation data with fetch, which Chrome blocks on file://, so the page is served on 127.0.0.1.
# The steps pick formula 399 (the smaller version tuned on the network, 8 particles): change '399' in the step files for another.
set -e
PAGE=$(cd "$1" && pwd); OUT=$(mkdir -p "$2" && cd "$2" && pwd); HERE=$(cd "$(dirname "$0")" && pwd); PORT=${PORT:-8799}
python3 -m http.server $PORT --bind 127.0.0.1 --directory "$PAGE" > /dev/null 2>&1 & SRV=$!; trap "kill $SRV" EXIT
until curl -s -o /dev/null "http://127.0.0.1:$PORT/index.html"; do sleep 0.2; done
node "$HERE/grab.js" "http://127.0.0.1:$PORT/index.html" "$HERE/steps_explain.json" "$OUT"
node "$HERE/grab.js" "http://127.0.0.1:$PORT/index.html" "$HERE/steps_tryjet.json" "$OUT"
python3 - "$OUT" <<'PY'
import sys
from PIL import Image
o = sys.argv[1]
for f, h in (('n10_groups', 760), ('n5_groups', 778), ('setup', 1000)):   # the slides show the top of these (their aspect ratio)
    im = Image.open(f'{o}/{f}.png'); im.crop((0, 0, im.width, min(h, im.height))).save(f'{o}/{f}_top.png')
PY
