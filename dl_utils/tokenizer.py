import string
import torch
import numpy as np

"""
The word "token" refers to an atomic string that has a direct mapping to
a token id. Token ids refer to an int id value that can then be used as
an index to an Embedding.

Use the tokenizer in the following way:

    X = [
        "hey I'm a string",
        "hey so am I",
    ]
    tokenizer = Tokenizer(strings=X)
    input_ids = tokenizer(
        strings=X,
        as_tensor=True,
        seq_len=128,
        add_bos=True,
        add_eos=True,)

or

    X = [
        "hey I'm a string",
        "hey so am I",
    ]
    tokenizer = Tokenizer()
    tokenizer.train(X=X)
    input_ids = tokenizer(
        strings=X,
        as_tensor=True,
        seq_len=128,
        add_bos=True,
        add_eos=True,)

or

    X = [
        ["hey", "I'm", "a", "list", "of", "strings",],
        ["hey", "so", "am", "I",],
    ]
    tokenizer = Tokenizer()
    tokenizer.train(tok_X=X)
    input_ids = tokenizer.toks_to_ids(
        toks=X,
        as_tensor=True,
        seq_len=128,
        add_bos=True,
        add_eos=True,)
    list_of_strs = tokenizer.decode(input_ids)
"""

def tokenize(main_string, delimeters={" "},
                          special_tokens={"\\newline",'\n'},
                          words=set(),
                          split_digits=False,
                          lowercase=False):
    """
    Returns a list of tokens delimeted by the strings contained in the
    delimeters set. Punctuation and whitespace characters are treated
    as individual tokens. Delimeters are not counted as tokens.

    Multiple delimeters in a row are counted as a single delimeter.

    Args:
        string: str
            Some string to be tokenized
        delimeters: set of str
            the delimeters to be used for tokenization
        special_tokens: set of str
            strings that should be treated as individual tokens. Cannot
            contain delimeting characters.
        split_digits: bool
            if true, strings of digits will be split into individual 
            digit tokens 0-9
        lowercase: bool
            if true, all tokens are lowercased
    Returns:
        tokens: list of str
    """
    tokens = []
    s = ""
    for i,char in enumerate(main_string):
        if s in special_tokens or s in words:
            tokens.append(s)
            s = ""
        
        if char in delimeters:
            if len(s) > 0:
                if split_digits and s.isdigit():
                    for c in s:
                        tokens.append(c)
                else:
                    tokens.append(s)
                    if lowercase:
                        tokens[-1] = tokens[-1].lower()
            s = ""
        elif split_digits and char.isdigit():
            if len(s) > 0:
                tokens.append(s)
            tokens.append(char)
            s = ""
        elif char.isalnum() or char=="_" or char=="<" or char==">":
            s += char
        elif char in string.whitespace:
            if len(s) > 0:
                if split_digits and s.isdigit():
                    for c in s:
                        tokens.append(c)
                else:
                    tokens.append(s)
                    if lowercase:
                        tokens[-1] = tokens[-1].lower()
            tokens.append(char)
            s = ""
        else:
            if len(s) > 0 and s[-1] != "\\":
                if split_digits and s.isdigit():
                    for c in s:
                        tokens.append(c)
                else:
                    tokens.append(s)
                    if lowercase:
                        tokens[-1] = tokens[-1].lower()
                s = ""
            if char == "\\":
                s += char
            else:
                tokens.append(char)
    if len(s) > 0:
        tokens.append(s)
    return tokens

def group_sentences(document, delimeters={".","!","?"},
                  titles={"dr","mr","mrs","ms","prof"}):
    """
    Groups a document string into a list of sentence strings.
    Sentences are defined by an argued delimeter followed by a
    whitespace character. Handles abbreviations by assuming all
    abbreviations are single, capital characters.

    Input:
        document: str
        delimeters: set of str
            end of sentence characters.
        titles: set of str
            enumerated titles that should be considered.
    Returns:
        sentences: list of str
            a list of all of the sentences in the document.
    """
    sentences = []
    running_s = document[0]
    document = document[1:]
    for i,c in enumerate(document[:-2]):
        running_s += c
        if c in delimeters:
            prob_dec = not document[i+1].isspace()
            other = (not document[i+2].isupper())
            other = other and not document[i-1].isalnum()
            prob_abbrev = document[i-1].isupper()
            prob_title =check_if_title(running_s[:-1],titles)
            if not(prob_dec or prob_abbrev or prob_title or other):
                running_s = running_s.strip()
                if len(running_s) > 0:
                    sentences.append(running_s)
                running_s = ""
    running_s = (running_s+document[-2:]).strip()
    if len(running_s) > 0:
        sentences.append(running_s)
    return sentences

def check_if_title(s, titles):
    """
    A helper function to check if the last word in the string is
    contained within a set of strings.

    s: str
        the string to determine if the last sequence of characters,
        delimeted by whitespace, is in the set `titles`
    titles: set of str
        the titles that need to be compared against
    """
    prob_title = False
    for ws_char in string.whitespace:
        splt = s.strip().split(ws_char)
        tid = -1 if len(splt) > 1 else 0
        title_str = splt[tid].strip().lower()
        prob_title = prob_title or title_str in titles
    return prob_title

def get_sent_arr(document, start_token="|<BOS>|",
                           stop_token="|<EOS>|",
                           delimeters={".","!","?"},
                           titles={"dr","mr","mrs","ms","prof"}):
    """
    Groups a document string into a list of sentence strings and then
    puts them into a matrix with a set sequence length.
    Sentences are defined by an argued delimeter followed by a
    whitespace character. Handles abbreviations by assuming all
    abbreviations are single, capital characters.

    Input:
        document: str
        seq_len: int or None
            if None is argued, the seq_len takes on the value of the
            longest sentence length in the document. Otherwise sentences
            that exceed the seq_len are broken into multiple segments
            each with a start and stop token at the start and end of
            the segments respectively.
        start_token: str
            the token value that should be placed at the beginning of
            each sentence
        stop_token: str
            the token value that should be placed at the end of
            each sentence
        delimeters: set of str
            end of sentence characters.
        titles: set of str
            enumerated titles that should be considered.
    Returns:
        tok_list: list of lists of str
            the result is a 2 dimensional matrix in which each entry in
            the row dimension is a sequence of tokens making up a
            sentence in that row.
    """
    sent_list = group_sentences(document,delimeters=delimeters,
                                         titles=titles)
    tok_list = []
    for i,sent in enumerate(sent_list):
        sent = start_token + " " + sent + " " + stop_token
        toks = tokenize(sent)
        tok_list.append(toks)
    return tok_list

class Tokenizer():
    pad_token = "<PAD>"
    bos_token = "<BOS>"
    eos_token = "<EOS>"
    unk_token = "<UNK>"
    """
    This class assists in tokenizing the data and converting between
    indices and tokens.
    """
    def __init__(self, word2id=None,
                       id2word=None,
                       split_digits=False,
                       pad_token="<PAD>",
                       bos_token="<BOS>",
                       eos_token="<EOS>",
                       unk_token="<UNK>",
                       strings=None,
                       words={"\\newline",'\n'},
                       delimeters={" "},
                       padding_side="right",
        ):
        """
        word2id: dict
            keys: str
                the words or tokens
            values: int
                the integer ids corresponding to each token
        id2word: dict
            keys: int
                the integer indices corresponding to each token
            values: str
                the words or tokens corresponding to each id
        split_digits: bool
            option to split each digit into a sequence of individual
            digits 0-9
        pad_token: str
            the padding token
        bos_token: str
            the beginning of sequence token
        eos_token: str
            the end of sequence token
        unk_token: str
            the token to correspond to the unknown embedding
        seq_len: int or None
            if None, then the maximum length of the tokenized X
            will be used for the X sequence length
        seq_len_y: int or None
            if None, then the maximum length of the tokenized Y
            will be used for the Y sequence length
        strings: list of str (optional)
            each string in this argued list is included in the
            conversion dictionaries word2id and id2word. Be careful,
            the argued words set will override the argued strings set.
            So, do not include subtrings of the argued strings in this
            set to the words set.
        words: set of str (optional)
            a set of words that should be included in the tokenization
        delimeters: set of str 
            strings to use as delimeters in the tokenization. if None,
            defaults to spaces.
        """
        self.padding_side = padding_side
        self.pad_token = Tokenizer.pad_token
        self.bos_token = Tokenizer.bos_token
        self.eos_token = Tokenizer.eos_token
        if pad_token and pad_token != Tokenizer.pad_token:
            self.pad_token = pad_token
        if bos_token and bos_token != Tokenizer.bos_token:
            self.bos_token = bos_token
        if eos_token and eos_token != Tokenizer.eos_token:
            self.eos_token = eos_token
        self.special_tokens = {
            "pad_token": self.pad_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
        }
        self.delimeters = delimeters
        if unk_token is not None:
            self.unk_token = unk_token
        else:
            self.unk_token = self.pad_token
        self.special_tokens["unk_token"] = self.unk_token
        self.split_digits = split_digits

        if words is None: words = set()
        #if split_digits: words |= set([str(i) for i in range(10)])
        words |= set(self.special_tokens.values())

        if word2id is None:
            word2id = {}
            id2word = {}

        for w in self.special_tokens.values():
            if w not in word2id:
                tid = len(word2id)
                word2id[w] = tid
                id2word[tid] = w

        # set special token id members (i.e. self.pad_id, self.bos_id,
        # self.eos_id)
        self.special_ids = dict()
        for k,v in self.special_tokens.items():
            splt = k.split("_")
            s = f"{splt[0]}_id"
            setattr(self, s, word2id[v])
            self.special_ids[s] = word2id[v]
            s = f"{splt[0]}_token_id"
            setattr(self, s, word2id[v])
            self.special_ids[s] = word2id[v]

        if strings is not None:
            for s in strings:
                if s not in word2id:
                    tid = len(word2id)
                    word2id[s] = tid
                    id2word[tid] = s
        for word in words:
            if word not in word2id:
                tid = len(word2id)
                word2id[word] = tid 
                id2word[tid] = word

        if id2word is None: id2word = dict()
        if word2id is None: word2id = dict()
        if len(word2id)!=len(id2word):
            word2id = {**{v:k for k,v in id2word.items()}, **word2id}
            id2word = {**{v:k for k,v in word2id.items()}, **id2word}
        self.word2id = word2id
        self.id2word = id2word

    @property
    def n_tokens(self):
        return len(self.word2id)

    def train(self,
              X=None,
              tok_X=[],
              alphabetize=False,
              verbose=True):
        """
        X: list of strings
            these are the strings that will be tokenized and indexed.
            if None is argued, the word2id and id2word must not be
            None
        tok_X: list of lists of tokens
            these are the strings that will be tokenized and indexed.
            if None is argued, the word2id and id2word must not be
            None
        alphabetize: bool
            if true, will alphabetize the token mapping, so that the
            words have ids according to their alphabetical ordering
        """
        words = set()
        assert len(tok_X) == 0 or X is None
        if X is not None:
            if verbose:
                print("Tokenizing")
            tok_X,_,new_words = self.tokenize(
                X, ret_all=True, verbose=verbose
            )
            words |= new_words
        else:
            if type(tok_X)==set: tok_X = list(tok_X)
            if type(tok_X[0])==str: tok_X = tok_X[0]
            for toks in tok_X:
                words |= set(toks)

        if alphabetize:
            words = sorted(list(words))
        for word in words:
            if word not in self.word2id:
                tid = len(self.word2id)
                self.word2id[word] = tid 
                self.id2word[tid] = word

    def convert2id(self, string, verbose=False):
        """
        This function is useful for getting the id of individual strings.

        Args:
            string: str
        Returns:
            id: int
        """
        try:
            return self.word2id[str(string)]
        except:
            if verbose:
                print(f"Tokenizer key error using {string}")
            return self.unk_id

    def convert_tokens_to_ids(self, string, verbose=False):
        return self.convert2id(string, verbose=verbose)

    def convert2word(self, id_, verbose=False):
        """
        This function is useful for getting the string of individual ids.

        Args:
            id: int
        Returns:
            string: str
        """
        try:
            return self.id2word[int(id_)]
        except:
            if verbose:
                print(f"Tokenizer error using {id_}")
            return self.unk_token

    def tokenize(self,
                 lostr,
                 ret_all=False,
                 special_tokens={"\\newline",'\n'},
                 verbose=False):
        """
        Args:
            lostr: list of str
                a list of strings to be tokenized
            ret_all: bool
                if false, returns only the token list
        Returns:
            toks: list of list of str (tokens)
                a list of lists of tokens
            max_len: int
                the maximum length of all the lists of tokens
            words: set
                a new set of words created from the tokenization process
        """
        special_tokens = {*special_tokens, *self.special_tokens.values()}
        max_len = 0
        toks = []
        words = set(self.word2id.keys())
        for i in range(len(lostr)):
            try:
                toks.append(
                    tokenize(
                        lostr[i],
                        split_digits=self.split_digits,
                        delimeters=self.delimeters,
                        special_tokens=special_tokens,
                        words=words,
                    )
                )
            except:
                print(i)
                print(lostr[i])
                assert False
            words |= set(toks[i])
            max_len = max(max_len,len(toks[i]))
            if verbose:
                print(round(float(i)/len(lostr)*100),"%", end="    \r")
        if ret_all:
            return toks,max_len,words
        return toks

    def toks_to_ids(self,
                    toks,
                    seq_len,
                    add_bos=False,
                    add_eos=False,
                    as_tensor=True,
                    truncation=False,
                    padding=None,
                    padding_side=None,
                    verbose=False):
        """
        Used to convert tokens to ids

        Args:
            toks: list of lists of str (N, variable)
                the tokens to be indexed
            seq_len: int
                the length of the sequence. if add_bos or add_eos is true,
                they will not add to this number. add_bos and add_eos will
                simply replace the tokens at the first and last locations
                in the sequence if there is not enough room for the whole
                sequence.
            add_bos: bool
                if true, self.bos_token is prepended to the start of the
                tokens.  will potentially overwrite last token if seq_len
                is not long enough to contain all tokens
            add_eos: bool
                if true, self.eos_token is appended to the end of the
                tokens will potentially overwrite last token if seq_len
                is not long enough to contain all tokens
            as_tensor: bool
                if true, will return a tensor, otherwise returns list
                of lists of ids
            truncation: bool
                if true, will truncate id lists longer than seq_len
            padding: None or string
                applies padding in different ways.

        Returns:
            X: list of lists of ids | torch long tensor (N,seq_len)
        """
        if not padding_side: padding_side = self.padding_side
        assert not as_tensor or (seq_len and as_tensor) 
        if seq_len is None: seq_len = np.inf
        assert seq_len>=int(add_eos)+int(add_bos)+1
        ids = []
        for i,samp in enumerate(toks):
            ids.append([])
            if add_bos: ids[i].append(self.bos_id)
            for j,t in enumerate(samp):
                if j==seq_len-1 and add_eos and truncation:
                    ids[i].append(self.eos_id)
                    break
                ids[i].append(self.convert2id(t, verbose=verbose))
            if add_eos and len(ids[i])<seq_len:
                ids[i].append(self.eos_id)
            if seq_len<np.inf and (len(ids[i])<seq_len or padding):
                pad_list = [self.pad_id for _ in range(seq_len-len(ids[i]))]
                if padding_side=="right":
                    ids[i] = ids[i] + pad_list
                else:
                    ids[i] = pad_list + ids[i]
            if seq_len<np.inf and truncation:
                ids[i] = ids[i][:seq_len]
            if verbose:
                print(round(float(i)/len(toks)*100),"%", end="    \r")

        if as_tensor:
            return torch.LongTensor(ids)
        return ids
    
    def ids_to_toks(self, ids, verbose=False):
        """
        Converts a list of token_ids to a list of lists of str

        Args:
            ids: list of lists of ints or tensor (B,S)
                the indices to be converted to string values
        Returns:
            strings: list of lists of str
                a list of lists of the string values of the argued
                ids
        """
        if type(ids)==int or (hasattr(ids, "shape") and len(ids.shape)==0):
            ids = [[ids]]
        elif hasattr(ids, "shape") and len(ids.shape)==1: ids = [ids]
        elif type(ids)==list and type(ids[0])==int: ids = [ids]
        toks = []
        for seq in ids:
            if len(seq)>0:
                s = []
                for i in seq:
                    s.append(self.convert2word(i, verbose=verbose))
                toks.append(s)
        return toks
    
    def ids_to_strs(self, ids, delimeter=" "):
        """
        Converts a list of indices to a list of stings

        Args:
            ids: int or list of ints or tensor
                the indices to be converted to string values
            delimeter: str
                the character to delimit the strings by.
        Returns:
            strings: list of str
                a list of the joined string values of the argued indices
        """
        toks = self.ids_to_toks(ids=ids)
        strings = []
        for seq in toks:
            strings.append(delimeter.join(seq))
        return strings
    
    def decode(self, ids):
        """
        Converts a list of indices to a list of stings

        Args:
            ids: int or list of ints or tensor
                the indices to be converted to string values
        Returns:
            strings: list of str
                a list of the joined string values of the argued indices
        """
        if type(ids) != torch.Tensor:
            if "pred_ids" in ids:
                ids = ids["pred_ids"]
            elif "logits" in ids:
                ids = torch.argmax(ids["logits"], dim=-1)
            elif hasattr(ids, logits):
                ids = torch.argmax(ids.logits, dim=-1)
        return self.ids_to_strs(ids)

    def batch_decode(self, ids):
        return self.decode(ids=ids)

    def strs_to_ids(self,
                    strings,
                    as_tensor=True,
                    seq_len=None,
                    add_eos=False,
                    add_bos=False,
                    truncation=False,
                    padding=True,
                    padding_side="right",
                    ):
        """
        Converts a list of strings to a list of token id lists or a
        single tensor of ids

        Args:
            strings: str or list of str
                the strings to be tokenized
            as_tensor: bool
                if true, will return indices as a pytorch long tensor
            seq_len: int or None
                optional argument to truncate/pad the indexes
            add_eos: bool
                if true, adds the eos token to the end of every
                string within strings
            add_bos: bool
                if true, adds the bos token to the beginning of every
                string within strings
            truncation: bool
                if true, will truncate id lists longer than seq_len
            padding: None or bool or string
                applies padding in different ways.
        Returns:
            ids: list of ints
                a list of the integer indices of each token in the
                argued strings
        """
        if not padding_side: padding_side = self.padding_side
        if type(strings)==str: strings = [strings]
        toks,max_len,_ = self.tokenize(
            strings,
            ret_all=True,
        )
        if seq_len is None and (as_tensor or padding):
            seq_len = max_len + add_eos + add_bos
        ids = self.toks_to_ids(
            toks,
            seq_len=seq_len,
            add_bos=add_bos,
            add_eos=add_eos,
            as_tensor=as_tensor,
            truncation=truncation,
            padding=padding,
            padding_side=padding_side,
        )
        return ids

    def __call__(self,
            strings,
            as_tensor=True,
            return_tensors=None,
            max_length=None,
            add_bos=True,
            add_eos=False,
            truncation=False,
            padding=True,
            padding_side=None,
        ):
        """
        Converts a list of strings to a list of tokens

        Args:
            strings: str or list of str
                the strings to be tokenized
            as_tensor: bool
                if true, will return indices as a pytorch long tensor
            max_length: int or None
                optional argument to truncate/pad the indexes
            add_bos: bool
                if true, adds the bos token to the start of every
                string within strings
            add_eos: bool
                if true, adds the eos token to the end of every
                string within strings
        Returns:
            ids: list of ints or LongTensor
                a list of the integer indices of each token in the
                argued strings
        """
        if not padding_side: padding_side = self.padding_side
        if return_tensors: as_tensor = True
        # I'll allow it
        if type(strings)==type(torch.zeros(0)):
            return self.ids_to_strs( strings )
        ids = self.strs_to_ids(
            strings,
            as_tensor=as_tensor,
            seq_len=max_length,
            add_eos=add_eos,
            add_bos=add_bos,
            truncation=truncation,
            padding=padding,
            padding_side=padding_side,
        )
        attention_mask = ids!=self.pad_id
        return {
            "input_ids": ids,
            "attention_mask": attention_mask,
        }


