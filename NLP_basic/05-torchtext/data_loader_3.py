#from torchtext import data
from torchtext.legacy import data
from torchtext.legacy.data import TabularDataset

class DataLoader(object):
    '''
    Data loader class to load text file using torchtext library.
    '''

    def __init__(
        self, train_fn,
        batch_size=64,
        valid_ratio=.2,
        device=-1,
        max_vocab=999999,
        min_freq=1,
        # whether to use end_of_token
        # bos, eos는 주로 NLG에서 필요하다. 
        use_eos=False,
        shuffle=True
    ):
        '''
        DataLoader initialization.
        :param train_fn: Train-set filename
        :param batch_size: Batchify data fot certain batch size.
        :param device: Device-id to load data (-1 for CPU)
        :param max_vocab: Maximum vocabulary size
        :param min_freq: Minimum frequency for loaded word.
        :param use_eos: If it is True, put <EOS> after every end of sentence.
        :param shuffle: If it is True, random shuffle the input data.
        '''
        super().__init__()

        # Define field of the input file.
        # The input file consists of two fields.
        
        self.label = data.Field(
            sequential=False,
            # 어느 클래스 있는지 세주면 좋지. 
            use_vocab=True,
            # 모르는 class 있으면 안돼. 
            unk_token=None
        )
        
        self.text = data.Field(
            use_vocab=True,
            batch_first=True,
            include_lengths=False,
            eos_token='<EOS>' if use_eos else None
        )

        # Those defined two columns will be delimited by TAB.
        # Thus, we use TabularDataset to load two columns in the input file.
        # We would have two separate input file: train_fn, valid_fn
        # Files consist of two columns: label field and text field.
        train, valid = TabularDataset(
            path=train_fn,
            format='tsv', 
            fields=[
                ('label', self.label),
                ('text', self.text),
            ],
        ).split(split_ratio=(1 - valid_ratio))

        # Those loaded dataset would be feeded into each iterator:
        # train iterator and valid iterator.
        # We sort input sentences by length, to group similar lengths.

        # dataset이 나왔으니깐, dataloader에 넣어주는 것. 
        self.train_loader, self.valid_loader = data.BucketIterator.splits(
            (train, valid),
            batch_size=batch_size,
            device='cuda:%d' % device if device >= 0 else 'cpu',
            shuffle=shuffle,
            # 비슷한 길이끼리, mini batch로 만들어라. 
            sort_key=lambda x: len(x.text),
            # mini-batch내에서 sorting을 해줄 것인가?
            # mini-batch내에서도 긴 애가 먼저 나오고, 짧은 애가 뒤에 나오게 할지를 의미. 
            sort_within_batch=True,
        )

        # At last, we make a vocabulary for label and text field.
        # It is making mapping table between words and indice.
        self.label.build_vocab(train)
        self.text.build_vocab(train, max_size=max_vocab, min_freq=min_freq)
