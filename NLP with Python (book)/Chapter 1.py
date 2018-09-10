import nltk
nltk.download()

from nltk.book import *

text1
text2

text1.concordance("monstrous")
text2.concordance("affection")
text3.concordance("lived")
text4.concordance("nation")
text4.concordance("terror")
text4.concordance("god")
text5.concordance("im")
text5.concordance("ur")
text5.concordance("lol")

text1.similar("monstrous")
text2.similar("monstrous")

text2.common_contexts(["monstrous","very"])

text4.dispersion_plot(["citizens","democracy","freedom","duties","America"])

#text3.generate("is") to be reinstated later

len(text3)

sorted(set(text3))
len(set(text3))

from _future_ import division
