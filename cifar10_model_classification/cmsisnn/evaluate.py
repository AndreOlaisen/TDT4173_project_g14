import sys
import pathlib
import collections
import subprocess


def get_most_likely(output):
    pairs = [l.strip().split(":") for l in output.strip().split("\n")]
    pairs = [(c, float(v)) for (c, v) in pairs]
    max_val = max(pairs, key=lambda p: p[1])[1]
    return [p for p in pairs if p[1] == max_val]


def evaluate_generated(program_path, img_path, img_label, count=1, save_path=None):
    img_path = pathlib.Path(img_path)
    if img_path.is_file():
        files = [img_path]
    else:
        files = list(img_path.iterdir())
    counter = collections.defaultdict(int)
    ties = 0
    for f in files[:min(count, len(files))]:
        args = [program_path, f]
        if save_path is not None:
            args.append(save_path/f"{f.stem}_act.json")
        process = subprocess.run(args, capture_output=True, text=True)
        process.check_returncode()
        most_likely = get_most_likely(process.stdout)
        for m, c in most_likely:
            counter[m] += c / len(most_likely)
        if len(most_likely) > 1:
            ties += 1
    return counter, ties
