import re
import nltk
import unidecode
import string

sent_tokenizer = nltk.data.load('tokenizers/punkt/portuguese.pickle')


stopwords = ["a", "à", "ah", "ai", "algo", "alguém", "algum", "alguma", "algumas",
             "alguns", "alô", "ambos", "ante", "ao", "apenas", "após", "aquela",
             "aquelas", "aquele", "aqueles", "aquilo", "as", "até", "bis", "cada",
             "certa", "certas", "certo", "certos", "chi", "com", "comigo", "conforme",
             "conosco", "consigo", "contigo", "contra", "convosco", "cuja", "cujas",
             "cujo", "cujos", "da", "das", "de", "dela", "delas", "dele", "deles",
                     "desde", "dessa", "dessas", "desse", "desses", "disso", "desta", "destas",
                     "deste", "destes", "disto", "daquela", "daquelas", "daquele", "daqueles",
                     "daquilo", "do", "dos", "e", "eia", "ela", "elas", "ele", "eles", "em",
                     "embora", "enquanto", "entre", "essa", "essas", "esse", "esses", "esta",
                     "este", "estes", "estou", "eu", "hem", "hum", "ih", "isso", "isto", "lhe",
                     "lhes", "logo", "mais", "mas", "me", "menos", "mesma", "mesmas", "mesmo",
                     "mesmos", "meu", "meus", "mim", "minha", "minhas", "muita", "muitas",
                     "muito", "muitos", "na", "nada", "nas", "nela", "nelas", "nele", "neles",
                     "nem", "nenhum", "nenhuma", "nenhumas", "nenhuns", "ninguém", "no", "nos",
                     "nós", "nossa", "nossas", "nosso", "nossos", "nela", "nelas", "nele",
                     "neles", "nessa", "nessas", "nesse", "nesses", "nisso", "nesta", "nestas",
                     "neste", "nestes", "nisto", "naquela", "naquelas", "naquele", "naqueles",
                     "naquilo", "o", "ó", "ô", "oba", "oh", "olá", "onde", "opa", "ora", "os",
                     "ou", "outra", "outras", "outrem", "outro", "outros", "para", "pelo",
                     "pela", "pelos", "pelas", "per", "perante", "pois", "por", "porém",
                     "porque", "portanto", "pouca", "poucas", "pouco", "poucos", "próprios",
                     "psit", "psiu", "quais", "quaisquer", "qual", "qualquer", "quando",
                     "quanta", "quantas", "quanto", "quantos", "que", "quem", "se", "sem",
                     "seu", "seus", "si", "sob", "sobre", "sua", "suas", "talvez", "tanta",
                     "tantas", "tanto", "tantos", "te", "teu", "teus", "ti", "toda", "todas",
                     "todo", "todos", "trás", "tu", "tua", "tuas", "tudo", "ué", "uh", "ui",
                     "um", "uma", "umas", "uns", "vária", "várias", "vário", "vários", "você",
                     "vós", "vossa", "vossas", "vosso", "vossos", "ser", "estar"]

abbrs = ['A', 'AA', 'abr', 'abrev', 'A.C', 'a.C', 'A/C', 'a/c', 'acad', 'A.D', 'adj', 'adm', 'aeron', 'ag', 'ag.to', 'ago', 'agr', 'agric', 'Al', 'alf', 'álg', 'ALM', 'alm', 'alt', 'alv', 'a.m', 'anat', 'ap', 'apart', 'arc', 'arcaic', 'arit', 'aritm', 'arq', 'arquit', 'art', 'ass.ª', 'assem', 'assemb', 'assist', 'assoc', 'astr', 'át', 'atm', 'at.te', 'aum', 'aut', 'auto', 'autom', 'aux', 'Av', 'av', 'aven', 'Bar', 'b.-art', 'b.-artes', 'B.el', 'B.eis', 'bibl', 'bibliog', 'bibliogr', 'bilog', 'bibliot', 'biofís', 'biogr', 'biol', 'bioq', 'bioquím', 'bisp', 'bispd', 'B.o', 'bomb', 'bot', 'br', 'bras', 'brasil', 'Brig', 'brig', 'Brig.o', 'brit', 'btl', 'bur', 'buroc', 'C', 'c', 'c/', 'c/a', 'c.c', 'c/c', 'C.-alm', 'c.-alm', 'cap', 'Cap.ão', 'cap.ão', 'capt', 'Cap', 'cap', 'caps', 'C.el', 'c.el', 'Cia', 'C.ia', 'ciênc', 'círc', 'cit', 'clim', 'climatol', 'col', 'com', 'com', 'comte', 'comte', 'comp', 'compl', 'cons', 'cons.º', 'consel', 'conselh', 'Const', 'const', 'const', 'constr', 'cont', 'contab', 'cos', 'cp', 'créd', 'cron', 'cronol', 'Cx', 'cx', 'D', 'D', 'd', 'Da', 'D.ª', 'D.C', 'd.C', 'DD', 'dec', 'decr', 'demog', 'demogr', 'Dep', 'dep', 'deps', 'des', 'desen', 'desc', 'dez', 'dez.º', 'dic', 'dipl', 'doc', 'docs', 'Dr', 'Drs', 'Dr.a', 'Dra', 'D.ra', 'Dr.as', 'E', 'e', 'E', 'EE', 'E.C', 'e.c.f', 'E.D', 'ed', 'ed', 'edif', 'ed', 'educ', 'e.g', 'elem', 'eletr', 'eletrôn', 'E.M', 'Em.ª', 'Emb', 'emb', 'Emb.or', 'embr', 'embriol', 'eng', 'enol', 'Esc', 'esp', 'equit', 'E.R', 'Est', 'est', 'etc', 'ex', 'Ex.ª', 'Ex.ma(o)', 'F', 'f', 'fáb', 'fac', 'farm', 'fed', 'fed', 'feder', 'fenôm', 'fev', 'fev.º', 'ff', 'filos', 'fin', 'fisc', 'Fl', 'folc', 'folcl', 'for', 'form', 'fot', 'foto', 'Fr', 'fss', 'fund', 'g', 'G.al', 'Gen', 'gen', 'gar', 'g.de', 'gen', 'geneal', 'geo', 'geog', 'geogr', 'ger', 'germ', 'gloss', 'G.M', 'g.m', 'g.-m', '', 'gov', 'G/P', 'gr', 'gráf', 'grav', 'h', 'hebr', 'her', 'herál', 'heráld', 'herd.o', 'hidr', 'hidrául', 'hig', 'hip', 'hist', 'humor', 'I', 'igr', 'ib', 'id', 'Il.ma(o)', 'imigr', 'impr', 'índ', 'ind', 'indúst', 'inform', 'jan', 'jorn', 'Jr', 'jud', 'jul', 'jun', 'jur', 'jur', 'juris', 'jurisp', 'jurispr', 'just. mil', 'J.z', 'L', 'l', 'L', 'l', 'l.º', 'lib', 'liv', 'livr', 'lab', 'laborat', 'lanç', 'larg', 'lat', 'L.da', 'L.do', 'Lic.do', 'leg', 'leg', 'legisl', 'm', 'm/', 'M.ª', 'm.ª', 'M', 'mun', 'mai', 'Maj', 'maj', 'maiúsc', 'M.al', 'Mal', 'mal', 'mar', 'mar', 'm.ço', 'mark', 'mat', 'matem', 'M.e', 'mec', 'mecân', 'med', 'méd', 'méd. vet', 'memo', 'memor', 'mens', 'met', 'metal', 'metalur', 'met', 'meteor', 'mil', 'miner', 'mit', 'mitol', 'MM', 'Mme', 'M.me', 'mme', 'mMin', 'mob', 'mod', 'moed', 'mon', 'monog', 'monogr', 'Mons', 'mons', 'mor', 'morf', 'morfol', 'm/p', 'm.to(a)', 'mus', 'mús', 'n', 'N. da D', 'N. da E', 'N. da R', 'N. do A', 'N. do D', 'N. do E', 'N. do T', 'nac', 'náut', 'naz', 'neol', 'N.S', 'N.Sr.a', 'nov', 'nov.o', 'N.T', 'num', 'núm', 'nº', 'o/', 'ob', 'obs', 'oc', 'ocid', 'odont', 'odontol', 'of', 'oft', 'ofalm', 'oftalmol', 'olig', 'ópt', 'or', 'orig', 'ord', 'org', 'organiz', 'ort', 'ortogr', '', 't', '', 't.o', 'p', 'p.a', 'p/', 'p', 'pag', 'pág', 'P', 'Pe', 'P.e', 'P.B', 'pc', 'pç',
'pç', 'pça', 'pal', 'pals', 'par', 'pat', 'patol', 'P.D', 'perf', 'p.ex', 'p. ext', 'abrev', 'pg', 'Ph.D', 'P.L', 'pl', 'p.m', 'poét', 'pol', 'polít', 'port', 'p.p', 'pq', 'Pr', 'Pres', 'pres', 'Presid', 'presid', 'proc', 'prod', 'Prof', 'prof', 'Prof.ª', 'prof.ª', 'Prof.as', 'prof.as', 'Profs', 'profs', 'pron', 'P.S', 'psic', 'psican', 'psic', 'psicol', 'PT', 'pt', 'q', 'q', 'q.b', 'q.do', 'Q.G', 'Q.-G', 'q.ta', 'q.to', 'quart', 'quest', 'quím', 'quinz', 'R', 'ref', 'reg', 'rg', 'rel', 'relat', 'rel', 'relig', 'Rem.te', 'rep', 'repúb', 'report', 'res', 'ret', 'retór', 'rev', 'Rev.mo', 'rod', 'R.S.V.P', 'rus', 'russ', 'S', 'S.A',
'Sarg', 's.d', 's/d', 'sec', 'secr', 'séc', 'sécs', 'segg', 'segs', 'ss', 'sem', 'sem', 'semin', 'S.Em.a', 'S.Em.as', 'serv', 'set', '', 'set.o', 'S.Ex.a', 'S.Ex.as', 'símb', 'Snr.a', 'Sr.a', 'S.O', 'Soc', 'soc', 'soc', 'sociol', 'Sr', 'S.res', 'Sr.es', 'sr.es', 'Sres', 'S.Rev.ma', 'S.rta', 'S.ta', 'Sr.ta', 'S.S', 'S.S.a', 'S.S.as', 'SS.Rev.mas', 'sub', 'Suc', 'surr', 'surreal', 'T', 'T', 'Trav', 'tb', 'teat', 'teatr', 'téc', 'técn', 'tecn', 'tecn', 'tecnol', 'tel', 'tele', 'telef', 'Ten', 'ten', 'T.te', 't.te', 'teol', 'terap', 'terapêut', 'tes', 'tip', 'tipogr', 'tít', 'topog', 'topogr', 'trad', 'transp', 'trig', 'trigon', 'trim', 'trop', 'tur', 'turism', 'u.e', 'u. e c', 'u.i', 'un', 'un', 'unif', 'univ', '', 'univ', 'univers', 'urb', 'urb', 'urban', 'urol', 'urug', 'us', 'util', 'utilid', 'util', 'utilit', 'utop', 'V', 'V', 'v', 'V.A', 'V.a', 'VV.AA', 'V.-Alm', 'v.-alm', 'V.E.ma', 'V.E.mas', 'vet', 'veter', 'V.Ex.a(s)', 'v.g', 'vin', 'vinic', 'V.M', 'voc', 'vol', 'vols', 'V.Rev.ma(s)', 'V.S', 'vs', 'V.S', 'V.S.a', 'V.S.as', 'VV.S.as', 'VV.SS', 'vulg', 'VV.MM', 'W.C', 'xenof', 'xerog', 'xerogr', 'zool', 'zoot', 'zootec']


sent_tokenizer._params.abbrev_types.update(abbrs)


def tokenize_sentences(text):
    return sent_tokenizer.tokenize(text)


def tokenize_paragraphs(text):
    return text.split("\n")


def tokenize_text(text):
    paragraphd = paragraph_tokenization(text)
    sentenced = [sentence_tokenization(paragraph)
                 for paragraph in paragraphd]

    return sentenced


def normalize(text, remove_stopwords=True, remove_punctuation=True, remove_accents=True, lower=True):
    if remove_punctuation:
        text = text.translate(str.maketrans('', '', string.punctuation))
    if remove_accents:
        text = unidecode.unidecode(text)
    if lower:
        text = text.lower()
    if remove_stopwords:
        tokens = nltk.tokenize.word_tokenize(text)
        tokens = [t for t in tokens if t not in stopwords]
        text = ' '.join(tokens)

    return text


def word_tokenization(sentence):
    return nltk.word_tokenize(sentence, language='portuguese')


def join_punctuation(match_object):
    return re.sub('\s+', '', match_object.group(0))


def posprocessing(text):
    # join punctuation
    text = re.sub(
        "\s+[\.\,\:\!\?\-\$\%\@\)\]\}]+", join_punctuation, text)
    text = text.replace('( ', '(').replace('[ ', '[').replace('{ ', '}')
    return text
