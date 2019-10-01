import pandas as pd
from joblib import load
from sklearn.naive_bayes import MultinomialNB

import re
import html





two_plus_letters_RE = re.compile(r"(\w)\1{1,}", re.DOTALL)			# regexp for word elongation: matches 3 or more repetitions of a word character
three_plus_letters_RE = re.compile(r"(\w)\1{2,}", re.DOTALL)
two_plus_words_RE = re.compile(r"(\w+\s+)\1{1,}", re.DOTALL)		# regexp for repeated words



##########################
# FUNCTION TO CLEAN TEXT #
##########################
def cleanup_text(text):
	text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', '', text)	# Remove URLs
	text = re.sub('@[^\s]+', '', text)								# Remove user mentions of the form @username
	if re.search("&#",text):										# Replace special html-encoded characters with their ASCII equivalent, for example: &#39 ==> '
		text = html.unescape(text)
	text = re.sub(r'_[xX]000[dD]_', '', text)						# Remove special useless characters such as _x000D_
	text = re.sub('[\W_]', ' ', text)								# Replace all non-word characters (such as emoticons, punctuation, end of line characters, etc.) with a space
	text = text.strip()												# Remove redundant white spaces
	text = re.sub('[\s]+', ' ', text)
	text = two_plus_letters_RE.sub(r"\1\1", text)					# normalize word elongations (characters repeated more than twice)
	text = two_plus_words_RE.sub(r"\1", text)						# remove repeated words
	text = re.sub("^\d+\s|\s\d+\s|\s\d+$", " ", text)				# remove any numbers inside the corpus "\d+"
	return text


#############################################
# FUNCTION TO DETECT LANGUAGE AND SENTIMENT #
#############################################
def detect_laguage(document):
	df = pd.DataFrame({'document':[], 'language':[]})
	# df.language = pd.Series(['TUN'])
	df.document = pd.Series(document)

	nb_model = load('exportSecondModel.joblib') 
	bow_model_char = load('exportBowModel.joblib')
	dtm_Test=bow_model_char.transform(df.document)
	fullmessage = document
	lang = str((nb_model.predict(dtm_Test))[0])
	match = float((nb_model.predict_proba(dtm_Test))[0][0])
	if (match < 0.55 and match > 0.47):
		lang = "OTHER"

	if (lang == "TUN"):
		sentimentarray = detect_sentiment_TUN(document)
	elif (lang == "ARA"):
		sentimentarray = detect_sentiment_ARA(document)
	else :
		sentimentarray = ['N/A', 999]
	return {'message': fullmessage,'lang': lang, 'match': match, 'sentiment': sentimentarray[0], 'sentimentmatch': sentimentarray [1]}



##############################
# FUNCTION TO READ TEXT FILE #
##############################
def read_text_file(filename):
	print('Reading file ' + filename + "...")
	with open(filename, "r", encoding='utf8') as textfile:
		L = []
		for line in textfile:
			L.append(line.strip())
		print('File contains ', len(L), "lines.\n")
		return L

##############################
# DETECT SENTIMENT OF ARABIC #
##############################
def detect_sentiment_ARA(document):
	df = pd.DataFrame({'document':[], 'language':[]})
	# df.language = pd.Series(['TUN'])
	df.document = pd.Series(document)

	bow_model = load('exportBowSentimentARA.joblib') 
	NB_model2 = load('exportModelSentimentARA.joblib') 

	dtm_Test=bow_model.transform(df.document)

	sentimentId = (NB_model2.predict(dtm_Test))[0]
	sentimentPercent = (NB_model2.predict_proba(dtm_Test))[0][0]
	if ( sentimentPercent < 0.55 and sentimentPercent > 0.47):
		sentiment = 'Neutral'
	elif (sentimentId == 1):
		sentiment = 'Positive'
	else:
		sentiment = 'Negative'
	return [ sentiment,sentimentPercent ]


################################
# DETECT SENTIMENT OF TUNISIAN #
################################
def detect_sentiment_TUN(document):
	df = pd.DataFrame({'document':[], 'language':[]})
	# df.language = pd.Series(['TUN'])
	df.document = pd.Series(document)

	bow_model_tun = load('exportBowSentimentTUN.joblib') 
	NB_model2 = load('exportModelSentimentTUN.joblib') 

	dtm_Test=bow_model_tun.transform(df.document)

	sentimentId = (NB_model2.predict(dtm_Test))[0]
	sentimentPercent = (NB_model2.predict_proba(dtm_Test))[0][0]
	if ( sentimentPercent < 0.55 and sentimentPercent > 0.47):
		sentiment = 'Neutral'
	elif (sentimentId == 1):
		sentiment = 'Positive'
	else:
		sentiment = 'Negative'
	return [ sentiment,sentimentPercent ]

