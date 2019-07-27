#!/usr/bin/env python3

'''
This code is based on as follows:
  * preprocessing: https://www.kaggle.com/christofhenkel/how-to-preprocessing-for-glove-part2-usage
  * training: https://www.kaggle.com/yuval6967/toxic-bert-plain-vanila
  * batch sampler: https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/discussion/94779
  * loss: https://www.kaggle.com/matsuik/subgroup-negative-weighting
  * evaluation: https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/discussion/90527
'''

#===========================================================
# train params
#===========================================================

VERSION = 1

PREPROCESSING = True
#PREPROCESSING = False

USE_META_FEATURES = False

N_JOBS = 28

train_params = {
    #'n_splits': 5,
    'n_epochs': 2,
    'lr': 2e-5,
    'warmup': 0.05,
    'weight_decay': 0.01,
    'batch_size': 64,
    'accumulation_steps': 1,
}


#===========================================================
# imports
#===========================================================

import gc
import os
import random
import re
import string
import time
from contextlib import contextmanager
from functools import partial
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from nltk.tokenize.treebank import TreebankWordTokenizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import BatchSampler, RandomSampler

from fastprogress import master_bar, progress_bar

# for GPT-2
from apex import amp
from pytorch_pretrained_bert import GPT2Tokenizer, OpenAIAdam
from pytorch_pretrained_bert import GPT2ClassificationHeadModel


#===========================================================
# global vars
#===========================================================

MAX_SEQUENCE_LENGTH = 220
SEED = 2019

ID_COLUMN = 'id'
COMMENT_COLUMN = 'comment_text'
TOXICITY_COLUMN = 'target'
AUX_COLUMNS  = ['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']
IDENTITY_COLUMNS = [
    'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
    'muslim', 'black', 'white', 'psychiatric_or_mental_illness'
]

JIGSAW_DIR = Path('../input')


#===========================================================
# utils
#===========================================================

@contextmanager
def timer(name):
    t0 = time.time()
    print(f'[{name}] start')
    yield
    print(f'[{name}] done in {time.time() - t0:.0f} s')


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class JigsawEvaluator():
    def __init__(self, y_true, y_identity, power=-5, overall_model_weight=0.25):
        self.y = y_true
        self.y_i = y_identity
        self.n_subgroups = self.y_i.shape[1]
        self.power = power
        self.overall_model_weight = overall_model_weight

    @staticmethod
    def _compute_auc(y_true, y_pred):
        try:
            return roc_auc_score(y_true, y_pred)
        except ValueError:
            return np.nan

    def _compute_subgroup_auc(self, i, y_pred):
        mask = self.y_i[:, i] == 1
        return self._compute_auc(self.y[mask], y_pred[mask])

    def _compute_bpsn_auc(self, i, y_pred):
        mask = self.y_i[:, i] + self.y == 1
        return self._compute_auc(self.y[mask], y_pred[mask])

    def _compute_bnsp_auc(self, i, y_pred):
        mask = self.y_i[:, i] + self.y != 1
        return self._compute_auc(self.y[mask], y_pred[mask])

    def compute_bias_metrics_for_model(self, y_pred):
        records = np.zeros((3, self.n_subgroups))
        for i in range(self.n_subgroups):
            records[0, i] = self._compute_subgroup_auc(i, y_pred)
            records[1, i] = self._compute_bpsn_auc(i, y_pred)
            records[2, i] = self._compute_bnsp_auc(i, y_pred)
        return records

    def _calculate_overall_auc(self, y_pred):
        return roc_auc_score(self.y, y_pred)

    def _power_mean(self, array):
        total = sum(np.power(array, self.power))
        return np.power(total / len(array), 1 / self.power)

    def get_final_metric(self, y_pred):
        bias_metrics = self.compute_bias_metrics_for_model(y_pred)
        bias_score = np.average([
            self._power_mean(bias_metrics[0]),
            self._power_mean(bias_metrics[1]),
            self._power_mean(bias_metrics[2])
        ])
        overall_score = self.overall_model_weight * self._calculate_overall_auc(y_pred)
        bias_score = (1 - self.overall_model_weight) * bias_score
        return overall_score + bias_score

    def get_all_score(self, y_pred):
        bias_metrics = self.compute_bias_metrics_for_model(y_pred)
        power_means = [
            self._power_mean(bias_metrics[0]),
            self._power_mean(bias_metrics[1]),
            self._power_mean(bias_metrics[2])
        ]
        bias_score = np.average(power_means)
        overall_auc = self._calculate_overall_auc(y_pred)
        overall_score = self.overall_model_weight * overall_auc
        bias_score = (1 - self.overall_model_weight) * bias_score
        return {
            'overall_auc': overall_auc,
            'subgroup_auc': power_means[0],
            'bpsn_auc': power_means[1],
            'bnsp_auc': power_means[2],
            'final_metrics': overall_score + bias_score,
        }


#===========================================================
# feature engineering
#===========================================================

contraction_patterns = [
    (b'US', b'United States'),
    (b'IT', b'Information Technology'),
    (b'(W|w)on\'t', b'will not'),
    (b'(C|c)an\'t', b'can not'),
    (b'(I|i)\'m', b'i am'),
    (b'(A|a)in\'t', b'is not'),
    (b'(\w+)\'ll', b'\g<1> will'),
    (b'(\w+)n\'t', b'\g<1> not'),
    (b'(\w+)\'ve', b'\g<1> have'),
    (b'(\w+)\'s', b'\g<1> is'),
    (b'(\w+)\'re', b'\g<1> are'),
    (b'(\w+)\'d', b'\g<1> would'),
]
patterns = [(re.compile(reg), repl) for (reg, repl) in contraction_patterns]


def count_regexp_occ(regexp='', text=None):
    return len(re.findall(regexp, text))


def prepare_for_char_n_gram(text):
    clean = bytes(text.lower(), encoding='utf-8')

    # drop \n and  \t
    clean = clean.replace(b'\n', b' ')
    clean = clean.replace(b'\t', b' ')
    clean = clean.replace(b'\b', b' ')
    clean = clean.replace(b'\r', b' ')

    # replace english contractions
    for (pattern, repl) in patterns:
        clean = re.sub(pattern, repl, clean)

    # drop puntuations
    exclude = re.compile(b'[%s]' % re.escape(bytes(string.punctuation, encoding='utf-8')))
    clean = b" ".join([exclude.sub(b'', token) for token in clean.split()])

    # drop numbers
    clean = re.sub(b'\d+', b' ', clean)

    # remove extra spaces
    clean = re.sub(b'\s+', b' ', clean)

    # remove ending space if any
    clean = re.sub(b'\s+$', b'', clean)

    # now replace words by words surrounded by # signs
    clean = re.sub(b' ', b'# #', clean)  # Replace space
    clean = b'#' + clean + b'#'          # add leading and trailing #

    return str(clean, 'utf-8')


def get_indicators_and_clean_comments(df):
    df_feat = pd.DataFrame(index=df['id'].index)

    with timer('  * basic word features'):
        # count number of \n
        df_feat['ant_slash_n'] = df[COMMENT_COLUMN].map(lambda x: count_regexp_occ(r"\n", x))
        # check number of upper case, if you're angry you may write in upper case
        df_feat['nb_upper'] = df[COMMENT_COLUMN].map(lambda x: count_regexp_occ(r"[A-Z]", x))
        # number of F words - f..k contains folk, fork,
        df_feat['nb_fk'] = df[COMMENT_COLUMN].map(lambda x: count_regexp_occ(r"[Ff]\S{2}[Kk]", x))
        # number of S word
        df_feat['nb_sk'] = df[COMMENT_COLUMN].map(lambda x: count_regexp_occ(r"[Ss]\S{2}[Kk]", x))
        # number of D words
        df_feat['nb_dk'] = df[COMMENT_COLUMN].map(lambda x: count_regexp_occ(r"[dD]ick", x))
        # number of occurence of You, insulting someone usually needs someone called : you
        df_feat['nb_you'] = df[COMMENT_COLUMN].map(lambda x: count_regexp_occ(r"\W[Yy]ou\W", x))
        # just to check you really refered to my mother ;-)
        df_feat['nb_mother'] = df[COMMENT_COLUMN].map(lambda x: count_regexp_occ(r"\Wmother\W", x))
        # just checking for toxic 19th century vocabulary
        df_feat['nb_ng'] = df[COMMENT_COLUMN].map(lambda x: count_regexp_occ(r"\Wnigger\W", x))
        # some Sentences start with a <:> so it may help
        df_feat['start_with_columns'] = df[COMMENT_COLUMN].map(lambda x: count_regexp_occ(r"^\:+", x))
        # check for time stamp
        df_feat['has_timestamp'] = df[COMMENT_COLUMN].map(lambda x: count_regexp_occ(r"\d{2}|:\d{2}", x))
        # check for dates 18:44, 8 December 2010
        df_feat['has_date_long'] = df[COMMENT_COLUMN].map(lambda x: count_regexp_occ(r"\D\d{2}:\d{2}, \d{1,2} \w+ \d{4}", x))
        # check for date short 8 December 2010
        df_feat['has_date_short'] = df[COMMENT_COLUMN].map(lambda x: count_regexp_occ(r"\D\d{1,2} \w+ \d{4}", x))
        # check for http links
        df_feat['has_http'] = df[COMMENT_COLUMN].map(lambda x: count_regexp_occ(r"http[s]{0,1}://\S+", x))
        # check for mail
        df_feat['has_mail'] = df[COMMENT_COLUMN].map(
            lambda x: count_regexp_occ(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', x)
        )
        # looking for words surrounded by == word == or """" word """"
        df_feat['has_emphasize_equal'] = df[COMMENT_COLUMN].map(lambda x: count_regexp_occ(r"\={2}.+\={2}", x))
        df_feat['has_emphasize_quotes'] = df[COMMENT_COLUMN].map(lambda x: count_regexp_occ(r"\"{4}\S+\"{4}", x))

        df_feat['chick_count'] = df[COMMENT_COLUMN].map(lambda x: x.count('!'))
        df_feat['qmark_count'] = df[COMMENT_COLUMN].map(lambda x: x.count('?'))

    with timer('  * clean comment features'):

        # clean comments
        df_feat['clean_comment'] = df[COMMENT_COLUMN].map(lambda x: prepare_for_char_n_gram(x))

        df_feat['clean_word_len'] = df_feat['clean_comment'].map(lambda x: len(x.split()))
        df_feat['clean_char_len'] = df_feat['clean_comment'].map(lambda x: len(x))
        df_feat['clean_chars'] = df_feat['clean_comment'].map(lambda x: len(set(x)))
        df_feat['clean_chars_ratio'] = df_feat['clean_comment'].map(lambda x: len(set(x))) / df_feat['clean_comment'].map(lambda x: 1 + min(99, len(x)))

        num_cols = [col for col in df_feat.columns if str(df_feat[col].dtype).count('int') or str(df_feat[col].dtype).count('float')]

    return df_feat[num_cols].values


#===========================================================
# preprocessing
#===========================================================

def set_mispell_compiler(spell_dict):
    return re.compile( '(%s)' % '|'.join(spell_dict.keys()))


first_mapping = {
    "_": " ",
    "`": "'",
    "’": "'",
    ";": "'",
    "‘": "'",
    "´": "'",
}

mispell_map = {
    "ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not",
    "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
    "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
    "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would", "i'd've":
    "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",
    "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam",
    "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have",
    "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have",
    "o'clock": "of the clock", "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not",
    "sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would", "she'd've": "she would have",
    "she'll": "she will", "she'll've": "she will have", "she's": "she is", "should've": "should have", "shouldn't": "should not",
    "shouldn't've": "should not have", "so've": "so have","so's": "so as", "this's":"this is","that'd": "that would",
    "that'd've": "that would have", "that's": "that is", "there'd": "there would", "there'd've": "there would have", "there's": "there is",
    "here's": "here is","they'd": "they would", "they'd've": "they would have", "they'll": "they will", "they'll've": "they will have",
    "they're": "they are", "they've": "they have", "to've": "to have", "wasn't": "was not", "we'd": "we would", "we'd've": "we would have",
    "we'll": "we will", "we'll've": "we will have", "we're": "we are", "we've": "we have", "weren't": "were not", "what'll": "what will",
    "what'll've": "what will have", "what're": "what are",  "what's": "what is", "what've": "what have", "when's": "when is",
    "when've": "when have", "where'd": "where did", "where's": "where is", "where've": "where have", "who'll": "who will", "who'll've": "who will have",
    "who's": "who is", "who've": "who have", "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not",
    "won't've": "will not have", "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have",
    "y'all": "you all", "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
    "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have", "you're": "you are", "you've": "you have",
    "trump's": "trump is", "obama's": "obama is", "canada's": "canada is", "today's": "today is",
    "He'd": "He would","He'll": "He will", "He's": "He is", "How'd": "How did", "How'd'y": "How do you", "How'll": "How will", "How's": "How is",
    "It'd've": "It would have", "It'll": "It will", "It'll've": "It will have","It's": "It is", "Let's": "Let us",
    "She'd": "She would", "She'd've": "She would have", "She'll": "She will", "She'll've": "She will have", "She's": "She is",
    "They're": "They are", "They've": "They have", "to've": "to have", "wasn't": "was not", "We'd": "We would", "We'd've": "We would have",
    "We'll": "We will", "We'll've": "We will have", "We're": "We are", "We've": "We have", "Weren't": "Were not", "What'll": "What will",
    "What'll've": "What will have", "What're": "What are",  "What's": "What is", "What've": "What have", "When's": "When is",
    "When've": "When have", "Where'd": "Where did", "Where's": "Where is", "Where've": "Where have", "Who'll": "Who will", "Who'll've": "Who will have",
    "Who's": "Who is", "Who've": "Who have", "Why's": "Why is", "Why've": "Why have",
    "You'd": "You would", "You'd've": "You would have", "You'll": "You will", "You'll've": "You will have", "You're": "You are", "You've": "You have",
    "Trump's": "Trump is", "Obama's": "Obama is", "Canada's": "Canada is"
}

first_compiler = set_mispell_compiler(first_mapping)
mispell_compiler = set_mispell_compiler(mispell_map)

symbols_to_isolate = '.,?!-;*"…:—()%#$&_/@＼・ω+=”“[]^–>\\°<~•≠™ˈʊɒ∞§{}·τα❤☺ɡ|¢→̶`❥━┣┫┗Ｏ►★©―ɪ✔®\x96\x92●£♥➤´¹☕≈÷♡◐║▬′ɔː€۩۞†μ✒➥═☆ˌ◄½ʻπδηλσερνʃ✬ＳＵＰＥＲＩＴ☻±♍µº¾✓◾؟．⬅℅»Вав❣⋅¿¬♫ＣＭβ█▓▒░⇒⭐›¡₂₃❧▰▔◞▀▂▃▄▅▆▇↙γ̄″☹➡«φ⅓„✋：¥̲̅́∙‛◇✏▷❓❗¶˚˙）сиʿ✨。ɑ\x80◕！％¯−ﬂﬁ₁²ʌ¼⁴⁄₄⌠♭✘╪▶☭✭♪☔☠♂☃☎✈✌✰❆☙○‣⚓年∎ℒ▪▙☏⅛ｃａｓǀ℮¸ｗ‚∼‖ℳ❄←☼⋆ʒ⊂、⅔¨͡๏⚾⚽Φ×θ￦？（℃⏩☮⚠月✊❌⭕▸■⇌☐☑⚡☄ǫ╭∩╮，例＞ʕɐ̣Δ₀✞┈╱╲▏▕┃╰▊▋╯┳┊≥☒↑☝ɹ✅☛♩☞ＡＪＢ◔◡↓♀⬆̱ℏ\x91⠀ˤ╚↺⇤∏✾◦♬³の｜／∵∴√Ω¤☜▲↳▫‿⬇✧ｏｖｍ－２０８＇‰≤∕ˆ⚜☁'
symbols_to_delete = '\n🍕\r🐵😑\xa0\ue014\t\uf818\uf04a\xad😢🐶️\uf0e0😜😎👊\u200b\u200e😁عدويهصقأناخلىبمغر😍💖💵Е👎😀😂\u202a\u202c🔥😄🏻💥ᴍʏʀᴇɴᴅᴏᴀᴋʜᴜʟᴛᴄᴘʙғᴊᴡɢ😋👏שלוםבי😱‼\x81エンジ故障\u2009🚌ᴵ͞🌟😊😳😧🙀😐😕\u200f👍😮😃😘אעכח💩💯⛽🚄🏼ஜ😖ᴠ🚲‐😟😈💪🙏🎯🌹😇💔😡\x7f👌ἐὶήιὲκἀίῃἴξ🙄Ｈ😠\ufeff\u2028😉😤⛺🙂\u3000تحكسة👮💙فزط😏🍾🎉😞\u2008🏾😅😭👻😥😔😓🏽🎆🍻🍽🎶🌺🤔😪\x08‑🐰🐇🐱🙆😨🙃💕𝘊𝘦𝘳𝘢𝘵𝘰𝘤𝘺𝘴𝘪𝘧𝘮𝘣💗💚地獄谷улкнПоАН🐾🐕😆ה🔗🚽歌舞伎🙈😴🏿🤗🇺🇸мυтѕ⤵🏆🎃😩\u200a🌠🐟💫💰💎эпрд\x95🖐🙅⛲🍰🤐👆🙌\u2002💛🙁👀🙊🙉\u2004ˢᵒʳʸᴼᴷᴺʷᵗʰᵉᵘ\x13🚬🤓\ue602😵άοόςέὸתמדףנרךצט😒͝🆕👅👥👄🔄🔤👉👤👶👲🔛🎓\uf0b7\uf04c\x9f\x10成都😣⏺😌🤑🌏😯ех😲Ἰᾶὁ💞🚓🔔📚🏀👐\u202d💤🍇\ue613小土豆🏡❔⁉\u202f👠》कर्मा🇹🇼🌸蔡英文🌞🎲レクサス😛外国人关系Сб💋💀🎄💜🤢َِьыгя不是\x9c\x9d🗑\u2005💃📣👿༼つ༽😰ḷЗз▱ц￼🤣卖温哥华议会下降你失去所有的钱加拿大坏税骗子🐝ツ🎅\x85🍺آإشء🎵🌎͟ἔ油别克🤡🤥😬🤧й\u2003🚀🤴ʲшчИОРФДЯМюж😝🖑ὐύύ特殊作戦群щ💨圆明园קℐ🏈😺🌍⏏ệ🍔🐮🍁🍆🍑🌮🌯🤦\u200d𝓒𝓲𝓿𝓵안영하세요ЖљКћ🍀😫🤤ῦ我出生在了可以说普通话汉语好极🎼🕺🍸🥂🗽🎇🎊🆘🤠👩🖒🚪天一家⚲\u2006⚭⚆⬭⬯⏖新✀╌🇫🇷🇩🇪🇮🇬🇧😷🇨🇦ХШ🌐\x1f杀鸡给猴看ʁ𝗪𝗵𝗲𝗻𝘆𝗼𝘂𝗿𝗮𝗹𝗶𝘇𝗯𝘁𝗰𝘀𝘅𝗽𝘄𝗱📺ϖ\u2000үսᴦᎥһͺ\u2007հ\u2001ɩｙｅ൦ｌƽｈ𝐓𝐡𝐞𝐫𝐮𝐝𝐚𝐃𝐜𝐩𝐭𝐢𝐨𝐧Ƅᴨןᑯ໐ΤᏧ௦Іᴑ܁𝐬𝐰𝐲𝐛𝐦𝐯𝐑𝐙𝐣𝐇𝐂𝐘𝟎ԜТᗞ౦〔Ꭻ𝐳𝐔𝐱𝟔𝟓𝐅🐋ﬃ💘💓ё𝘥𝘯𝘶💐🌋🌄🌅𝙬𝙖𝙨𝙤𝙣𝙡𝙮𝙘𝙠𝙚𝙙𝙜𝙧𝙥𝙩𝙪𝙗𝙞𝙝𝙛👺🐷ℋ𝐀𝐥𝐪🚶𝙢Ἱ🤘ͦ💸ج패티Ｗ𝙇ᵻ👂👃ɜ🎫\uf0a7БУі🚢🚂ગુજરાતીῆ🏃𝓬𝓻𝓴𝓮𝓽𝓼☘﴾̯﴿₽\ue807𝑻𝒆𝒍𝒕𝒉𝒓𝒖𝒂𝒏𝒅𝒔𝒎𝒗𝒊👽😙\u200cЛ‒🎾👹⎌🏒⛸公寓养宠物吗🏄🐀🚑🤷操美𝒑𝒚𝒐𝑴🤙🐒欢迎来到阿拉斯ספ𝙫🐈𝒌𝙊𝙭𝙆𝙋𝙍𝘼𝙅ﷻ🦄巨收赢得白鬼愤怒要买额ẽ🚗🐳𝟏𝐟𝟖𝟑𝟕𝒄𝟗𝐠𝙄𝙃👇锟斤拷𝗢𝟳𝟱𝟬⦁マルハニチロ株式社⛷한국어ㄸㅓ니͜ʖ𝘿𝙔₵𝒩ℯ𝒾𝓁𝒶𝓉𝓇𝓊𝓃𝓈𝓅ℴ𝒻𝒽𝓀𝓌𝒸𝓎𝙏ζ𝙟𝘃𝗺𝟮𝟭𝟯𝟲👋🦊多伦🐽🎻🎹⛓🏹🍷🦆为和中友谊祝贺与其想象对法如直接问用自己猜本传教士没积唯认识基督徒曾经让相信耶稣复活死怪他但当们聊些政治题时候战胜因圣把全堂结婚孩恐惧且栗谓这样还♾🎸🤕🤒⛑🎁批判检讨🏝🦁🙋😶쥐스탱트뤼도석유가격인상이경제황을렵게만들지않록잘관리해야합다캐나에서대마초와화약금의품런성분갈때는반드시허된사용🔫👁凸ὰ💲🗯𝙈Ἄ𝒇𝒈𝒘𝒃𝑬𝑶𝕾𝖙𝖗𝖆𝖎𝖌𝖍𝖕𝖊𝖔𝖑𝖉𝖓𝖐𝖜𝖞𝖚𝖇𝕿𝖘𝖄𝖛𝖒𝖋𝖂𝕴𝖟𝖈𝕸👑🚿💡知彼百\uf005𝙀𝒛𝑲𝑳𝑾𝒋𝟒😦𝙒𝘾𝘽🏐𝘩𝘨ὼṑ𝑱𝑹𝑫𝑵𝑪🇰🇵👾ᓇᒧᔭᐃᐧᐦᑳᐨᓃᓂᑲᐸᑭᑎᓀᐣ🐄🎈🔨🐎🤞🐸💟🎰🌝🛳点击查版🍭𝑥𝑦𝑧ＮＧ👣\uf020っ🏉ф💭🎥Ξ🐴👨🤳🦍\x0b🍩𝑯𝒒😗𝟐🏂👳🍗🕉🐲چی𝑮𝗕𝗴🍒ꜥⲣⲏ🐑⏰鉄リ事件ї💊「」\uf203\uf09a\uf222\ue608\uf202\uf099\uf469\ue607\uf410\ue600燻製シ虚偽屁理屈Г𝑩𝑰𝒀𝑺🌤𝗳𝗜𝗙𝗦𝗧🍊ὺἈἡχῖΛ⤏🇳𝒙ψՁմեռայինրւդձ冬至ὀ𝒁🔹🤚🍎𝑷🐂💅𝘬𝘱𝘸𝘷𝘐𝘭𝘓𝘖𝘹𝘲𝘫کΒώ💢ΜΟΝΑΕ🇱♲𝝈↴💒⊘Ȼ🚴🖕🖤🥘📍👈➕🚫🎨🌑🐻𝐎𝐍𝐊𝑭🤖🎎😼🕷ｇｒｎｔｉｄｕｆｂｋ𝟰🇴🇭🇻🇲𝗞𝗭𝗘𝗤👼📉🍟🍦🌈🔭《🐊🐍\uf10aლڡ🐦\U0001f92f\U0001f92a🐡💳ἱ🙇𝗸𝗟𝗠𝗷🥜さようなら🔼'

isolate_dict = {ord(c):f' {c} ' for c in symbols_to_isolate}
remove_dict = {ord(c):f'' for c in symbols_to_delete}


def replace_typo(text, spell_dict, compiler):
    def replace(match):
        return spell_dict[match.group(0)]
    return compiler.sub(replace, text)


def handle_punctuation(x):
    x = x.translate(remove_dict)
    x = x.translate(isolate_dict)
    return x


def handle_contractions(x, tokenizer):
    x = tokenizer.tokenize(x)
    return x


def fix_quote(x):
    x = [x_[1:] if x_.startswith("'") else x_ for x_ in x]
    x = ' '.join(x)
    return x


def preprocess(x):
    tokenizer=TreebankWordTokenizer()
    x = replace_typo(x, first_mapping, first_compiler)
    x = replace_typo(x, mispell_map, mispell_compiler)
    x = handle_punctuation(x)
    x = handle_contractions(x, tokenizer=tokenizer)
    x = fix_quote(x)
    return x


def parallel_clean(ids, comments):
    return [[i, preprocess(x)] for i, x in zip(ids, comments)]


#===========================================================
# tokenization
#===========================================================

def convert_lines(example, max_seq_length, tokenizer):
    max_seq_length -= 2
    all_tokens = []
    longer = 0

    #for text in progress_bar(example):
    for text in example:
        tokens_a = tokenizer.tokenize(text)
        if len(tokens_a) > max_seq_length:
            tokens_a = tokens_a[:max_seq_length]
            longer += 1
        one_token = tokenizer.convert_tokens_to_ids(tokens_a) + [0] * (max_seq_length - len(tokens_a))
        all_tokens.append(one_token)

    return np.array(all_tokens)


#===========================================================
# model
#===========================================================

class CustomGPT2Classifier(nn.Module):
    def __init__(self, n_meta=0):
        super().__init__()
        gpt2_model = GPT2ClassificationHeadModel.from_pretrained('gpt2', clf_dropout=0.4, n_class=1)

        self.gpt2 = gpt2_model.transformer
        self.dropout = gpt2_model.dropout

        self.n_meta = n_meta
        if n_meta > 0:
            self.linear_meta = nn.Linear(n_meta, n_meta)

        self.linear_out = nn.Linear(768 * 2 + n_meta, 1)
        self.linear_aux_out = nn.Linear(768 * 2 + n_meta, 6)

    def forward(self, input_ids, x_meta=None, position_ids=None, token_type_ids=None, lm_labels=None, past=None):
        hidden_states, presents = self.gpt2(input_ids, position_ids, token_type_ids, past)
        avg_pool = torch.mean(hidden_states, 1)
        max_pool, _ = torch.max(hidden_states, 1)
        h_conc = torch.cat([avg_pool, max_pool], 1)

        h_conc = self.dropout(h_conc)

        if self.n_meta > 0:
            meta_output = self.linear_meta(x_meta)
            conc = torch.cat([h_conc, meta_output], 1)
        else:
            conc = h_conc

        result = self.linear_out(conc)
        aux_result = self.linear_aux_out(conc)

        out = torch.cat([result, aux_result], 1)

        return out


#===========================================================
# loss
#===========================================================

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        bce_loss = F.binary_cross_entropy_with_logits(input, target, reduction='none', weight=self.weight)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss

        return focal_loss.mean()


def custom_loss(data, targets, loss_weight):
    loss_1 = FocalLoss(weight=targets[:,1:2])(data[:,:1],targets[:,:1])
    loss_2 = nn.BCEWithLogitsLoss()(data[:,1:],targets[:,2:])

    return (loss_1 * loss_weight) + loss_2


#===========================================================
# batch sampler
#===========================================================

class LenMatchBatchSampler(BatchSampler):
    def __iter__(self):
        buckets = [[]] * 100
        yielded = 0

        for idx in self.sampler:
            count_zeros = torch.sum(self.sampler.data_source[idx][0] == 0)
            count_zeros = int(count_zeros / 64)
            if len(buckets[count_zeros]) == 0:  buckets[count_zeros] = []

            buckets[count_zeros].append(idx)

            if len(buckets[count_zeros]) == self.batch_size:
                batch = list(buckets[count_zeros])
                yield batch
                yielded += 1
                buckets[count_zeros] = []

        batch = []
        leftover = [idx for bucket in buckets for idx in bucket]

        for idx in leftover:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yielded += 1
                yield batch
                batch = []

        if len(batch) > 0 and not self.drop_last:
            yielded += 1
            yield batch

        assert len(self) == yielded, f'produced an inccorect number of batches. expected {len(self)}, but yielded {yielded}'


def trim_tensors(tsrs):
    max_len = torch.max(torch.sum((tsrs[0] != 0), 1))
    if max_len > 2:
        tsrs = [tsr[:, :max_len] for tsr in tsrs]
    return tsrs


#===========================================================
# train
#===========================================================

def train(x_train, x_train_meta, y_train, y_identity_train, all_ids, params, loss_func):
    n_epochs = params['n_epochs']
    batch_size = params['batch_size']
    lr = params['lr']
    warmup = params['warmup']
    weight_decay = params['weight_decay']
    accumulation_steps = params['accumulation_steps']

    train_dataset = TensorDataset(torch.tensor(x_train, dtype=torch.long),
                                  torch.tensor(x_train_meta, dtype=torch.float),
                                  torch.tensor(y_train, dtype=torch.float))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    if USE_META_FEATURES:
        model = CustomGPT2Classifier(n_meta=x_train_meta.shape[1])
    else:
        model = CustomGPT2Classifier(n_meta=0)
    model.zero_grad()
    model.cuda()

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    num_train_optimization_steps = int(n_epochs * len(train_dataset) / batch_size / accumulation_steps)
    optimizer = OpenAIAdam(optimizer_grouped_parameters,
                           lr=lr,
                           warmup=warmup,
                           schedule='warmup_cosine',
                           t_total=num_train_optimization_steps)

    model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=0)

    mb = master_bar(range(n_epochs))
    start_time = time.time()
    model.train()

    for epoch in mb:
        lossf = None

        optimizer.zero_grad()

        for i, (x_batch, x_meta_batch, y_batch) in enumerate(progress_bar(train_loader, parent=mb)):
            y_pred = model(x_batch.cuda())

            loss = loss_func(y_pred, y_batch.cuda())
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()

            # gradient accumulation
            if (i+1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            if lossf:
                lossf = 0.98 * lossf + 0.02 * loss.item()
            else:
                lossf = loss.item()

        torch.save(model.state_dict(), f'./model/gpt2_full_epoch{epoch+1}_seed{SEED}_v{VERSION}.bin')


def main():
    seed_torch(SEED)
    Path('./model').mkdir(exist_ok=True)

    if PREPROCESSING:
        Path('./cache').mkdir(exist_ok=True)

        with timer('Load train csv'):
            train_df = pd.read_csv(JIGSAW_DIR / 'train.csv')
            train_ids = train_df[ID_COLUMN].values

            train_size = len(train_df)
            chunk_size = train_size // N_JOBS
            parallel_indices = []

            for n in range(N_JOBS):
                if n == N_JOBS - 1:
                    parallel_indices.append((n * chunk_size, None))
                else:
                    parallel_indices.append((n * chunk_size, (n+1) * chunk_size))

        with timer('Feature engineering'):
           train_feature = get_indicators_and_clean_comments(train_df)
           joblib.dump(train_feature, './cache/train_feature.pkl')

        with timer('Preprocessing'):
            train_comments = train_df[COMMENT_COLUMN].values
            result = Parallel(n_jobs=N_JOBS)(
                    [delayed(parallel_clean)(train_ids[i1:i2],
                                             train_comments[i1:i2]) for (i1, i2) in parallel_indices]
            )

            dfs = []
            for tmp in result:
                df = pd.DataFrame(tmp, columns=[ID_COLUMN, COMMENT_COLUMN])
                dfs.append(df)

            merged_df = pd.concat(dfs, axis=0)
            merged_df.sort_values(by=ID_COLUMN, ascending=True, inplace=True)

            clean_text = merged_df[COMMENT_COLUMN].values
            train_df[COMMENT_COLUMN] = clean_text

            del result, dfs, merged_df
            gc.collect()

            joblib.dump(train_df, './cache/train_df_preprocessed.pkl')

        with timer('Tokenization'):
            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

            comments = train_df[COMMENT_COLUMN].fillna('DUMMY_VALUE')
            convert_lines_para = partial(convert_lines,
                                         max_seq_length=MAX_SEQUENCE_LENGTH,
                                         tokenizer=tokenizer)

            # XXX: parallel processing cause some error
            result = Parallel(n_jobs=1)(
                    [delayed(convert_lines_para)(comments[i1:i2]) for (i1, i2) in parallel_indices]
            )

            sequences = np.concatenate(result)
            joblib.dump(sequences, './cache/gpt2_sequences.pkl')

            del tokenizer, comments, result
            gc.collect()
    else:
        with timer('Load preprocessed data'):
            train_df = joblib.load('./cache/train_df_preprocessed.pkl')
            sequences = joblib.load('./cache/gpt2_sequences.pkl')
            train_feature = joblib.load('./cache/train_feature.pkl')

            train_ids = train_df[ID_COLUMN].values

    with timer('Feature scaling (meta features)'):
        scaler = StandardScaler()
        train_feature = scaler.fit_transform(train_feature)
        joblib.dump(scaler, './cache/scaler.pkl')

    with timer('Setup target values'):
        # subgorup negative weighting
        subgroup_bool_train = train_df[IDENTITY_COLUMNS].fillna(0) >= 0.5
        toxic_bool_train = train_df[TOXICITY_COLUMN].fillna(0) >= 0.5
        subgroup_negative_mask = subgroup_bool_train.values.sum(axis=1).astype(bool) & ~toxic_bool_train

        weights = np.ones((len(train_df),))
        weights += subgroup_negative_mask
        loss_weight = 1.0 / weights.mean()

        y_train = np.vstack([(train_df[TOXICITY_COLUMN].values >= 0.5).astype(np.int), weights]).T
        y_identity_train = np.where(train_df[IDENTITY_COLUMNS] >= 0.5, True, False)
        y_aux_train = train_df[AUX_COLUMNS].values

        y_train_all = np.hstack([y_train, y_aux_train])

        del train_df, y_train, y_aux_train, weights
        gc.collect()

    criterion = partial(custom_loss, loss_weight=loss_weight)
    train(sequences, train_feature, y_train_all, y_identity_train, train_ids, train_params, criterion)


if __name__ == '__main__':
    main()

