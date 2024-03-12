import locale
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import NllbTokenizer
from tqdm.auto import tqdm, trange
import re
import sys
import typing as tp
import unicodedata
from sacremoses import MosesPunctNormalizer
from transformers import AutoModelForSeq2SeqLM
from transformers import NllbTokenizer
import gc
import random
import numpy as np
import torch
from tqdm.auto import tqdm, trange
from transformers.optimization import Adafactor
from transformers import get_constant_schedule_with_warmup


LANGS = [('spanish', 'rus_Cyrl'), ('quechua', 'tyv_qu')]

def cleanup():
    """Try to free GPU memory"""
    gc.collect()
    torch.cuda.empty_cache()

def gpe(x=None):
    return "UTF-8"
    

def word_tokenize(text):
    # a very naive word tokenizer for languages with English-like orthography
    return re.findall('(\w+|[^\w\s])', text)



def get_non_printing_char_replacer(replace_by: str = " ") -> tp.Callable[[str], str]:
    non_printable_map = {
        ord(c): replace_by
        for c in (chr(i) for i in range(sys.maxunicode + 1))
        # same as \p{C} in perl
        # see https://www.unicode.org/reports/tr44/#General_Category_Values
        if unicodedata.category(c) in {"C", "Cc", "Cf", "Cs", "Co", "Cn"}
    }

    def replace_non_printing_char(line) -> str:
        return line.translate(non_printable_map)

    return replace_non_printing_char

def preproc(text):
    clean = mpn.normalize(text)
    clean = replace_nonprint(clean)
    clean = unicodedata.normalize("NFKC", clean)
    return clean

def fix_tokenizer(tokenizer, new_lang='tyv_qu'):
    """
    Add a new language token to the tokenizer vocabulary
    (this should be done each time after its initialization)
    """
    old_len = len(tokenizer) - int(new_lang in tokenizer.added_tokens_encoder)
    tokenizer.lang_code_to_id[new_lang] = old_len-1
    tokenizer.id_to_lang_code[old_len-1] = new_lang
    # always move "mask" to the last position
    tokenizer.fairseq_tokens_to_ids["<mask>"] = len(tokenizer.sp_model) + len(tokenizer.lang_code_to_id) + tokenizer.fairseq_offset

    tokenizer.fairseq_tokens_to_ids.update(tokenizer.lang_code_to_id)
    tokenizer.fairseq_ids_to_tokens = {v: k for k, v in tokenizer.fairseq_tokens_to_ids.items()}
    if new_lang not in tokenizer._additional_special_tokens:
        tokenizer._additional_special_tokens.append(new_lang)
    # clear the added token encoder; otherwise a new token may end up there by mistake
    tokenizer.added_tokens_encoder = {}
    tokenizer.added_tokens_decoder = {}


def get_batch_pairs(batch_size, data=df_train):
    (l1, long1), (l2, long2) = random.sample(LANGS, 2)
    xx, yy = [], []
    for _ in range(batch_size):
        item = data.iloc[random.randint(0, len(data)-1)]
        xx.append(preproc(item[l1]))
        yy.append(preproc(item[l2]))
    return xx, yy, long1, long2


def main():
    locale.getpreferredencoding = gpe
    trans_df = pd.read_excel('sp_qu_mt_corpus.xlsx')
    train_dev, df_test = train_test_split(trans_df, test_size=0.10, random_state=42)
    df_train, df_dev = train_test_split(train_dev, test_size=0.111, random_state=42)
    tokenizer = NllbTokenizer.from_pretrained('facebook/nllb-200-distilled-600M')
    smpl = df_train.sample(5653, random_state=1)
    smpl['qu_toks'] = smpl.quechua.apply(tokenizer.tokenize)
    smpl['sp_toks'] = smpl.spanish.apply(tokenizer.tokenize)
    smpl['qu_words'] = smpl.quechua.apply(word_tokenize)
    smpl['sp_words'] = smpl.spanish.apply(word_tokenize)
    mpn = MosesPunctNormalizer(lang="en")
    mpn.substitutions = [(re.compile(r), sub) for r, sub in mpn.substitutions]
    replace_nonprint = get_non_printing_char_replacer(" ")
    texts_with_unk_normed = [text for text in tqdm(texts_with_unk) if tokenizer.unk_token_id in tokenizer(preproc(text)).input_ids]
    tokenizer = NllbTokenizer.from_pretrained('facebook/nllb-200-distilled-600M')
    fix_tokenizer(tokenizer)
    model = AutoModelForSeq2SeqLM.from_pretrained('facebook/nllb-200-distilled-600M')
    model.resize_token_embeddings(len(tokenizer))
    model.model.shared.weight.data[added_token_id+1] = model.model.shared.weight.data[added_token_id]
    model.model.shared.weight.data[added_token_id] = model.model.shared.weight.data[similar_lang_id]
    cleanup()
    model.cuda();
    optimizer = Adafactor(
        [p for p in model.parameters() if p.requires_grad],
        scale_parameter=False,
        relative_step=False,
        lr=1e-4,
        clip_threshold=1.0,
        weight_decay=1e-3,
    )
    batch_size = 32  # 32 already doesn't fit well to 15GB of GPU memory
    max_length = 128
    warmup_steps = 1_000
    training_steps = 57000
    losses = []
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
    model.train()
    x, y, loss = None, None, None
    cleanup()

    tq = trange(len(losses), training_steps)
    for i in tq:
        xx, yy, lang1, lang2 = get_batch_pairs(batch_size)
        try:
            tokenizer.src_lang = lang1
            x = tokenizer(xx, return_tensors='pt', padding=True, truncation=True, max_length=max_length).to(model.device)
            tokenizer.src_lang = lang2
            y = tokenizer(yy, return_tensors='pt', padding=True, truncation=True, max_length=max_length).to(model.device)
            y.input_ids[y.input_ids == tokenizer.pad_token_id] = -100

            loss = model(**x, labels=y.input_ids).loss
            loss.backward()
            losses.append(loss.item())

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        except RuntimeError as e:
            optimizer.zero_grad(set_to_none=True)
            x, y, loss = None, None, None
            cleanup()
            print('error', max(len(s) for s in xx + yy), e)
            continue

        if i % 1000 == 0:
            print(i, np.mean(losses[-1000:]))

        if i % 1000 == 0 and i > 0:
            model.save_pretrained("save")
            tokenizer.save_pretrained("save")
    LANGS = [('s', 'rus_Cyrl'), ('quechua', 'tyv_qu')]
    
    
main()