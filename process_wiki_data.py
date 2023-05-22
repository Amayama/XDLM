# This script will generate .src and .tgt files, totalling 6 files
import datasets
import tqdm
import stanza
stanza.download('en')
nlp=stanza.Pipeline('en')
# Load the WMT14 dataset and select the training split
# splits = ['train']
splits = ['train','validation','test']

for split in splits:
    wmt14 = datasets.load_dataset('wikipedia', '20220301.en',split=split)

    if split == 'validation':
        split = 'dev'
    # Create a file to write the preprocessed data to
    with open(split+'.src', 'w', encoding='utf-8') as f:
        
        # Loop over each sample in the training split
        #TODO:  改成全量
        for sample in tqdm.tqdm(range(10000)):
        
            preprocessed_src = wmt14[sample]['text'].strip().replace('\n', '').replace('\r', '')
            doc=nlp(preprocessed_src)
            preprocessed_src=[sentence.text for sentence in doc.sentences]
            for sent in preprocessed_src:
                if split == 'train':
                    if len(sent)>=10:
                        f.write(sent + '\n')
                else:
                    if len(sent)>=2:
                        f.write(sent + '\n')
    print ('Done..',split)