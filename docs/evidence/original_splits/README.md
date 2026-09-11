# Original split files, as submitted (September 2025)

These are the `ImageSets/Main` files exactly as the dissertation's experiments used them. They are
kept because they are the evidence for finding 2 in `../../AUDIT.md`: they were generated per
dataset root by an unseeded `random.shuffle`, so the roots disagree about which images are held
out, and 219 of the 242 ids in `line2voc/val.txt` also appear in
`line2voc_preprocessed/train.txt`.

The splits under `data/` have since been regenerated as one seeded stratified split shared by all
three roots. Anything that needs to reproduce the original numbers should read from here.
