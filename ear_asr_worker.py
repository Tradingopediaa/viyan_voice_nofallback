import argparse
import contextlib
import json
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio")
    ap.add_argument("--lang", default=None)
    ap.add_argument("--model-card", default="omniASR_CTC_1B_v2")
    ap.add_argument("--import-only", action="store_true")
    args = ap.parse_args()

    with contextlib.redirect_stdout(sys.stderr):
        from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

    if args.import_only:
        print(json.dumps({"ok": True, "import": "omnilingual_asr"}))
        return

    if not args.audio:
        print(json.dumps({"ok": False, "error": "missing --audio"}))
        return

    with contextlib.redirect_stdout(sys.stderr):
        pipe = ASRInferencePipeline(model_card=args.model_card)
        try:
            if args.lang:
                out = pipe.transcribe([args.audio], lang=[args.lang], batch_size=1)
            else:
                out = pipe.transcribe([args.audio], batch_size=1)
        except TypeError:
            out = pipe.transcribe([args.audio], batch_size=1)

    text = out[0] if isinstance(out, list) and out else out
    print(json.dumps({
        "ok": True,
        "text": text,
        "model_card": args.model_card,
        "lang": args.lang
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
