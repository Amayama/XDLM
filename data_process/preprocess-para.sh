OUTPATH=data/processed/para  # path where processed files will be stored
FASTBPE=tools/fastBPE/fast  # path to the fastBPE tool

# create output path
mkdir -p $OUTPATH

# learn bpe codes on the training set (or only use a subset of it)
$FASTBPE learnbpe 50000 data/para/de-en.en.train > $OUTPATH/codes
$FASTBPE learnbpe 50000 data/para/de-en.de.train > $OUTPATH/codes

$FASTBPE applybpe $OUTPATH/de-en.train.en data/para/de-en.train.en $OUTPATH/codes &
$FASTBPE applybpe $OUTPATH/de-en.test.en data/para/de-en.test.en $OUTPATH/codes &
$FASTBPE applybpe $OUTPATH/de-en.valid.en data/para/de-en.valid.en $OUTPATH/codes &
$FASTBPE applybpe $OUTPATH/de-en.train.de data/para/de-en.train.de $OUTPATH/codes &
$FASTBPE applybpe $OUTPATH/de-en.test.de data/para/de-en.test.de $OUTPATH/codes &
$FASTBPE applybpe $OUTPATH/de-en.valid.de data/para/de-en.valid.de $OUTPATH/codes &

cat $OUTPATH/de-en.en.train | $FASTBPE getvocab - > $OUTPATH/vocab &
cat $OUTPATH/de-en.de.train | $FASTBPE getvocab - > $OUTPATH/vocab &
cp vocab dict.en.txt
cp vocab dict.de.txt
pair=de-en

for lg in $(echo $pair | sed -e 's/\-/ /g'); do
  for split in train valid test; do
    $FASTBPE applybpe $OUTPATH/$pair.$lg.$split data/para/$pair.$lg.$split $OUTPATH/codes
    python preprocess.py $OUTPATH/vocab $OUTPATH/$pair.$lg.$split
  done
done
